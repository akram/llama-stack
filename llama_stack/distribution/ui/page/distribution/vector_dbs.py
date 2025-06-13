# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from modules.api import llama_stack_api


def vector_dbs():
    st.header("Vector Databases")
    vector_dbs_info = {v.identifier: v.to_dict() for v in llama_stack_api.client.vector_dbs.list()}

    if len(vector_dbs_info) > 0:
        selected_vector_db = st.selectbox("Select a vector database", list(vector_dbs_info.keys()))
        st.json(vector_dbs_info[selected_vector_db])
        
        # Add delete button with confirmation
        if st.button("Delete Selected Vector Database", type="primary"):
            # Show confirmation dialog
            if st.session_state.get("confirm_delete") is None:
                st.session_state.confirm_delete = True
                st.warning(f"Are you sure you want to delete '{selected_vector_db}'? This action cannot be undone.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete"):
                        try:
                            llama_stack_api.client.vector_dbs.delete(selected_vector_db)
                            st.success(f"Successfully deleted vector database '{selected_vector_db}'")
                            # Clear the confirmation state
                            st.session_state.confirm_delete = None
                            # Force a rerun to update the list
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting vector database: {str(e)}")
                with col2:
                    if st.button("No, Cancel"):
                        st.session_state.confirm_delete = None
                        st.rerun()
    else:
        st.info("No vector databases found")
