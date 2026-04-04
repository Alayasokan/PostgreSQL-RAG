import streamlit as st
from db import Tags

st.title("Manage Tags")

def delete_tag(tag_id: int):
    Tags.delete().where(Tags.id == tag_id).execute()

@st.dialog("Add tag")
def add_tag_dialog_open():
    tag = st.text_input("Tag")
    if tag and st.button("Add", key="add-tag-button"):
        Tags.create(name=tag)
        st.rerun()

st.button("Add Tag", key="add-tag-button", on_click=add_tag_dialog_open)

tags = Tags.select()

if tags.exists():
    for tag in tags:
        with st.container(border=True):
            tag_name_col, _, delete_button_col = st.columns(3, vertical_alignment="center")
            with tag_name_col:
                st.write(tag.name)
            with delete_button_col:
                st.button("Delete", key=f"delete-tag-button-{tag.id}", on_click=lambda tid=tag.id: delete_tag(tid))
else:
    st.info("No tags created yet. Please create one!")