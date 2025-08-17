import os
import streamlit as st
from dotenv import load_dotenv

from app.ingest import ingest_folder
from app.rag import answer

def main():
    st.set_page_config(page_title="RAG Customer Support Assistant", page_icon="💬")
    st.title("💬 RAG Customer Support Assistant")
    st.markdown("Index your docs in `data/` then ask questions. Answers include inline citations like `[source: file#chunk]`.")
    load_dotenv()

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("Rebuild Index"):
            with st.spinner("Indexing documents..."):
                try:
                    ingest_folder(data_dir="data", storage_dir="storage")
                    st.success("Index rebuilt successfully.")
                except Exception as e:
                    st.error(f"Failed to index: {e}")
        st.markdown("**Models** are read from `.env` (see `.env.example`).")

    question = st.text_input("Ask a question about your docs:", placeholder="What's your return policy?")
    k = st.slider("Top-K context chunks", min_value=3, max_value=10, value=5, step=1)

    if question:
        with st.spinner("Thinking..."):
            try:
                result = answer(question, k=k, storage_dir="storage")
                st.markdown("### Answer")
                st.write(result["answer"])
                with st.expander("View retrieved context chunks"):
                    for c in result["contexts"]:
                        st.markdown(f"**{c['source']}#chunk-{c['chunk_id']}**")
                        st.code(c["text"][:1200])
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("Tip: Drop your FAQs, policies, and manuals into the `data/` folder, then click **Rebuild Index**.")

if __name__ == "__main__":
    main()
