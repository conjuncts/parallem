ParquetWriter
ParquetUniqueWriter
cast_document_to_bytes
store_input
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
- [ ] batch API but where a response has an error (in other words, test _decode_gemini_batch_error)
- [ ] decode_batch_content (this should be easily testable using real data)


- test max_batch_size
- test Batch API (and all other - integration test) by comparing batch output but not sending it
- test out max_batch_size (! convenience point!)
