# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, Mock, patch

import pytest
import pytest_asyncio
import yaml
from pydantic import TypeAdapter, ValidationError

from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import ModelType
from llama_stack.distribution.access_control.access_control import AccessDeniedError, is_action_allowed
from llama_stack.distribution.datatypes import AccessRule, ModelWithOwner, User
from llama_stack.distribution.routing_tables.models import ModelsRoutingTable


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def _return_model(model):
    return model


@pytest_asyncio.fixture
async def test_setup(cached_disk_dist_registry):
    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=cached_disk_dist_registry,
        policy={},
    )
    yield cached_disk_dist_registry, routing_table


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_access_control_with_cache(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-public",
        provider_id="test_provider",
        provider_resource_id="model-public",
        model_type=ModelType.llm,
    )
    model_admin_only = ModelWithOwner(
        identifier="model-admin",
        provider_id="test_provider",
        provider_resource_id="model-admin",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"]}),
    )
    model_data_scientist = ModelWithOwner(
        identifier="model-data-scientist",
        provider_id="test_provider",
        provider_resource_id="model-data-scientist",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["data-scientist", "researcher"], "teams": ["ml-team"]}),
    )
    await registry.register(model_public)
    await registry.register(model_admin_only)
    await registry.register(model_data_scientist)

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["admin"], "teams": ["management"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2

    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-admin")
    assert model.identifier == "model-admin"
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-data-scientist")

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["data-scientist"], "teams": ["other-team"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public"
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-admin")
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-data-scientist")

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["data-scientist"], "teams": ["ml-team"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2
    model_ids = [m.identifier for m in all_models.data]
    assert "model-public" in model_ids
    assert "model-data-scientist" in model_ids
    assert "model-admin" not in model_ids
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-data-scientist")
    assert model.identifier == "model-data-scientist"
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-admin")


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_access_control_and_updates(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-updates",
        provider_id="test_provider",
        provider_resource_id="model-updates",
        model_type=ModelType.llm,
    )
    await registry.register(model_public)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["user"],
        },
    )
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"
    model_public.owner = User("testuser", {"roles": ["admin"]})
    await registry.update(model_public)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["user"],
        },
    )
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-updates")
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["admin"],
        },
    )
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_access_control_empty_attributes(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model = ModelWithOwner(
        identifier="model-empty-attrs",
        provider_id="test_provider",
        provider_resource_id="model-empty-attrs",
        model_type=ModelType.llm,
        owner=User("testuser", {}),
    )
    await registry.register(model)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": [],
        },
    )
    result = await routing_table.get_model("model-empty-attrs")
    assert result.identifier == "model-empty-attrs"
    all_models = await routing_table.list_models()
    model_ids = [m.identifier for m in all_models.data]
    assert "model-empty-attrs" in model_ids


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_no_user_attributes(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-public-2",
        provider_id="test_provider",
        provider_resource_id="model-public-2",
        model_type=ModelType.llm,
    )
    model_restricted = ModelWithOwner(
        identifier="model-restricted",
        provider_id="test_provider",
        provider_resource_id="model-restricted",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"]}),
    )
    await registry.register(model_public)
    await registry.register(model_restricted)
    mock_get_authenticated_user.return_value = User("test-user", None)
    model = await routing_table.get_model("model-public-2")
    assert model.identifier == "model-public-2"

    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-restricted")

    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public-2"


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_automatic_access_attributes(mock_get_authenticated_user, test_setup):
    """Test that newly created resources inherit access attributes from their creator."""
    registry, routing_table = test_setup

    # Set creator's attributes
    creator_attributes = {"roles": ["data-scientist"], "teams": ["ml-team"], "projects": ["llama-3"]}
    mock_get_authenticated_user.return_value = User("test-user", creator_attributes)

    # Create model without explicit access attributes
    model = ModelWithOwner(
        identifier="auto-access-model",
        provider_id="test_provider",
        provider_resource_id="auto-access-model",
        model_type=ModelType.llm,
    )
    await routing_table.register_object(model)

    # Verify the model got creator's attributes
    registered_model = await routing_table.get_model("auto-access-model")
    assert registered_model.owner is not None
    assert registered_model.owner.attributes is not None
    assert registered_model.owner.attributes["roles"] == ["data-scientist"]
    assert registered_model.owner.attributes["teams"] == ["ml-team"]
    assert registered_model.owner.attributes["projects"] == ["llama-3"]

    # Verify another user without matching attributes can't access it
    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["engineer"], "teams": ["infra-team"]})
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("auto-access-model")

    # But a user with matching attributes can
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["data-scientist", "engineer"],
            "teams": ["ml-team", "platform-team"],
            "projects": ["llama-3"],
        },
    )
    model = await routing_table.get_model("auto-access-model")
    assert model.identifier == "auto-access-model"


@pytest_asyncio.fixture
async def test_setup_with_access_policy(cached_disk_dist_registry):
    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    mock_inference.unregister_model = AsyncMock(side_effect=_return_model)

    config = """
                - permit:
                    principal: user-1
                    actions: [create, read, delete]
                    description: user-1 has full access to all models
                - permit:
                    principal: user-2
                    actions: [read]
                    resource: model::model-1
                    description: user-2 has read access to model-1 only
                - permit:
                    principal: user-3
                    actions: [read]
                    resource: model::model-2
                    description: user-3 has read access to model-2 only
                - forbid:
                    actions: [create, read, delete]
             """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=cached_disk_dist_registry,
        policy=policy,
    )
    yield routing_table


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_access_policy(mock_get_authenticated_user, test_setup_with_access_policy):
    routing_table = test_setup_with_access_policy
    mock_get_authenticated_user.return_value = User(
        "user-1",
        {
            "roles": ["admin"],
            "projects": ["foo", "bar"],
        },
    )
    await routing_table.register_model("model-1", provider_id="test_provider")
    await routing_table.register_model("model-2", provider_id="test_provider")
    await routing_table.register_model("model-3", provider_id="test_provider")
    model = await routing_table.get_model("model-1")
    assert model.identifier == "model-1"
    model = await routing_table.get_model("model-2")
    assert model.identifier == "model-2"
    model = await routing_table.get_model("model-3")
    assert model.identifier == "model-3"

    mock_get_authenticated_user.return_value = User(
        "user-2",
        {
            "roles": ["user"],
            "projects": ["foo"],
        },
    )
    model = await routing_table.get_model("model-1")
    assert model.identifier == "model-1"
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-2")
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-3")
    with pytest.raises(AccessDeniedError):
        await routing_table.register_model("model-4", provider_id="test_provider")
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("model-1")

    mock_get_authenticated_user.return_value = User(
        "user-3",
        {
            "roles": ["user"],
            "projects": ["bar"],
        },
    )
    model = await routing_table.get_model("model-2")
    assert model.identifier == "model-2"
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-1")
    with pytest.raises(AccessDeniedError):
        await routing_table.get_model("model-3")
    with pytest.raises(AccessDeniedError):
        await routing_table.register_model("model-5", provider_id="test_provider")
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("model-2")

    mock_get_authenticated_user.return_value = User(
        "user-1",
        {
            "roles": ["admin"],
            "projects": ["foo", "bar"],
        },
    )
    await routing_table.unregister_model("model-3")
    with pytest.raises(ValueError):
        await routing_table.get_model("model-3")


def test_permit_when():
    config = """
    - permit:
        principal: user-1
        actions: [read]
      when: user in owners namespaces
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert not is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_permit_unless():
    config = """
    - permit:
        principal: user-1
        actions: [read]
        resource: model::*
      unless:
        - user not in owners namespaces
        - user in owners teams
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert not is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_forbid_when():
    config = """
    - forbid:
        principal: user-1
        actions: [read]
      when:
        user in owners namespaces
    - permit:
        actions: [read]
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_forbid_unless():
    config = """
    - forbid:
        principal: user-1
        actions: [read]
      unless:
        user in owners namespaces
    - permit:
        actions: [read]
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_user_has_attribute():
    config = """
    - permit:
        actions: [read]
      when: user with admin in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_user_does_not_have_attribute():
    config = """
    - permit:
        actions: [read]
      unless: user with admin not in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_is_owner():
    config = """
    - permit:
        actions: [read]
      when: user is owner
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("user-2", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_is_not_owner():
    config = """
    - permit:
        actions: [read]
      unless: user is not owner
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("user-2", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_invalid_rule_permit_and_forbid_both_specified():
    config = """
    - permit:
        actions: [read]
      forbid:
        actions: [create]
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_rule_neither_permit_or_forbid_specified():
    config = """
    - when: user is owner
      unless: user with admin in roles
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_rule_when_and_unless_both_specified():
    config = """
    - permit:
        actions: [read]
      when: user is owner
      unless: user with admin in roles
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_condition():
    config = """
    - permit:
        actions: [read]
      when: random words that are not valid
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


@pytest.mark.parametrize(
    "condition",
    [
        "user is owner",
        "user is not owner",
        "user with dev in teams",
        "user with default not in namespaces",
        "user in owners roles",
        "user not in owners projects",
    ],
)
def test_condition_reprs(condition):
    from llama_stack.distribution.access_control.conditions import parse_condition

    assert condition == str(parse_condition(condition))


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_valid_user_access_empty_list_scenarios(mock_get_authenticated_user, test_setup):
    """Test different scenarios where users should get empty lists vs 403 errors."""
    registry, routing_table = test_setup

    # Scenario 1: Empty registry - users with general access should get empty list
    # Note: Users without general access should get 403 even with empty registry (tested separately)
    mock_get_authenticated_user.return_value = User("any-user", {"roles": ["user"], "teams": ["team1"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 0
    assert all_models.data == []

    # Scenario 2: User has access but no matching resources exist
    # Add a public model that everyone can access
    model_public = ModelWithOwner(
        identifier="model-public",
        provider_id="test_provider",
        provider_resource_id="model-public",
        model_type=ModelType.llm,
        # No owner means public access
    )
    await registry.register(model_public)

    # User should see the public model
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public"

    # Scenario 3: Test that users with proper access get empty lists when no resources match
    # This should be different from the 403 case - if a user has read access but there are
    # no resources that match their specific attributes, they should get empty list

    # Add a model that only specific users can access
    model_restricted = ModelWithOwner(
        identifier="model-restricted",
        provider_id="test_provider",
        provider_resource_id="model-restricted",
        model_type=ModelType.llm,
        owner=User("owner1", {"roles": ["admin"], "teams": ["admin-team"]}),
    )
    await registry.register(model_restricted)

    # Remove the public model so we only have restricted model
    await registry.delete("model", "model-public")

    # Now test the key scenario: user has general access but no resources match their attributes
    # This should be the scenario that returns empty list instead of 403

    # This is the case that should return empty list, not 403 - when user has some access
    # but there are no resources that match their specific attributes
    mock_get_authenticated_user.return_value = User("valid-user", {"roles": ["user"], "teams": ["team1"]})

    # This is currently failing but should pass if implemented correctly
    # The user should get empty list because they have valid access but no matching resources
    try:
        all_models = await routing_table.list_models()
        assert len(all_models.data) == 0
        assert all_models.data == []
        print("✓ Test passed: User with valid access gets empty list when no resources match")
    except AccessDeniedError:
        print("✗ Test failed: User with valid access got 403 instead of empty list")
        # This is the current behavior that we want to test for
        raise AssertionError(
            "Expected empty list, got 403 error - this indicates the access control behavior needs refinement"
        ) from None


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_access_denied_for_users_with_no_general_access(
    mock_get_authenticated_user, test_setup_with_access_policy
):
    """Test that users with no general access get 403 errors in all scenarios."""
    routing_table = test_setup_with_access_policy

    # Set up a specific user who should have no access according to the policy
    mock_get_authenticated_user.return_value = User(
        "user-no-access",
        {
            "roles": ["restricted"],
            "projects": ["unauthorized"],
        },
    )

    # Case 1: Empty registry - users with no general access should get 403, not empty list
    with pytest.raises(AccessDeniedError):
        await routing_table.list_models()

    # Case 2: Registry with resources - users with no general access should still get 403
    # Register a model as an admin user first
    mock_get_authenticated_user.return_value = User("user-1", {"roles": ["admin"], "projects": ["foo", "bar"]})
    await routing_table.register_model("test-model", provider_id="test_provider")

    # Now switch back to the restricted user
    mock_get_authenticated_user.return_value = User(
        "user-no-access",
        {
            "roles": ["restricted"],
            "projects": ["unauthorized"],
        },
    )

    # This user should still get 403 because they have no general access
    with pytest.raises(AccessDeniedError):
        await routing_table.list_models()


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_delete_access_denied_returns_403_not_404(mock_get_authenticated_user, test_setup_with_access_policy):
    """Test that delete operations return 403 instead of 404 when user lacks access to existing resource."""
    routing_table = test_setup_with_access_policy

    # Register a model as user-1 (who has full access according to the policy)
    mock_get_authenticated_user.return_value = User("user-1", {"roles": ["admin"], "projects": ["foo", "bar"]})
    await routing_table.register_model("restricted-model", provider_id="test_provider")

    # Switch to a user with no access to the model (not user-1, user-2, or user-3)
    mock_get_authenticated_user.return_value = User(
        "no-access-user",
        {
            "roles": ["restricted"],
            "projects": ["unauthorized"],
        },
    )

    # Try to get the object - should get 403 access denied, not 404 not found
    with pytest.raises(AccessDeniedError):
        await routing_table.get_object_by_identifier("model", "restricted-model")

    # This means delete operations will also properly return 403 instead of 404
    # since they use get_object_by_identifier internally


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_idempotent_delete_behavior(mock_get_authenticated_user, test_setup_with_access_policy):
    """Test that delete operations are idempotent when user has permission."""
    routing_table = test_setup_with_access_policy

    # Set up user-1 who has DELETE permission
    mock_get_authenticated_user.return_value = User("user-1", {"roles": ["admin"], "projects": ["foo", "bar"]})

    # Test 1: Delete non-existent resource should succeed (idempotent)
    # This should NOT raise an error, even though the resource doesn't exist
    await routing_table.unregister_model("non-existent-model")  # Should succeed

    # Test 2: Delete existing resource should succeed
    await routing_table.register_model("test-model", provider_id="test_provider")
    await routing_table.unregister_model("test-model")  # Should succeed

    # Test 3: Delete the same resource again should succeed (idempotent)
    await routing_table.unregister_model("test-model")  # Should succeed again

    # Test 4: User without DELETE permission should get 403 regardless of resource existence
    mock_get_authenticated_user.return_value = User(
        "no-delete-user",
        {
            "roles": ["restricted"],
            "projects": ["unauthorized"],
        },
    )

    # Should get 403 for non-existent resource
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("non-existent-model")

    # Should get 403 for existing resource (create one as user-1 first)
    mock_get_authenticated_user.return_value = User("user-1", {"roles": ["admin"], "projects": ["foo", "bar"]})
    await routing_table.register_model("protected-model", provider_id="test_provider")

    mock_get_authenticated_user.return_value = User(
        "no-delete-user",
        {
            "roles": ["restricted"],
            "projects": ["unauthorized"],
        },
    )

    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("protected-model")


@pytest.mark.asyncio
@patch("llama_stack.distribution.routing_tables.common.get_authenticated_user")
async def test_vector_db_delete_behavior(mock_get_authenticated_user, test_setup_with_access_policy):
    """Test vector DB delete operations match the user's requirements."""
    routing_table = test_setup_with_access_policy

    # Note: This test uses ModelsRoutingTable, but the logic is the same for VectorDBsRoutingTable
    # since they both inherit from CommonRoutingTableImpl

    # Test 1: User with DELETE permission deleting non-existent resource → 200 OK (idempotent)
    mock_get_authenticated_user.return_value = User("user-1", {"roles": ["admin"], "projects": ["foo", "bar"]})

    # This should succeed without any error (idempotent delete)
    await routing_table.unregister_model("my_demo_vector_db")

    # Test 2: User with DELETE permission deleting existing resource → 200 OK
    await routing_table.register_model("my_demo_vector_db", provider_id="test_provider")
    await routing_table.unregister_model("my_demo_vector_db")  # Should succeed

    # Test 3: User with DELETE permission deleting already-deleted resource → 200 OK (idempotent)
    await routing_table.unregister_model("my_demo_vector_db")  # Should succeed again

    # Test 4: User WITHOUT DELETE permission → 403 Forbidden (regardless of resource existence)
    mock_get_authenticated_user.return_value = User(
        "no-delete-user", {"roles": ["restricted"], "projects": ["unauthorized"]}
    )

    # Should get 403 for non-existent resource
    with pytest.raises(AccessDeniedError) as exc_info:
        await routing_table.unregister_model("my_demo_vector_db")

    # Verify it's a proper 403 access denied error
    assert "cannot perform action 'delete'" in str(exc_info.value)
