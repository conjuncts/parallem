ParquetWriter
ParquetUniqueWriter
cast_document_to_str
store_doc_hash
batch(tools, structured_output)(gemini, openai)
images

msg_state bug where ask_llm(documents="One single string") would break down the string into many items
msg_state.ask_functions()
- [ ] If there are pending but not submitted batch calls, they are automatically sent
- [ ] Same for pending calls
- [ ] See if we can use SQLite's upsert
- [ ] msg_state.resolve()
- [ ] msg_state.ask_human()
- [ ] tools.auto_schema
- [ ] ask_params
- [ ] mcp server

- [ ] error example: image/ppm type
