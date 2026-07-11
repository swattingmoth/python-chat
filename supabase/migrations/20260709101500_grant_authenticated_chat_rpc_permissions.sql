-- Allow authenticated users to use chat persistence RPCs.
-- Functions are SECURITY INVOKER, so caller needs underlying table/sequence privileges.

begin;

grant usage on schema public to authenticated;

grant select, insert, update on public.chat_sessions to authenticated;
grant select, insert on public.chat_messages to authenticated;
grant select, insert on public.tool_calls to authenticated;

grant usage, select on sequence public.chat_sessions_id_seq to authenticated;
grant usage, select on sequence public.chat_messages_id_seq to authenticated;
grant usage, select on sequence public.tool_calls_id_seq to authenticated;

grant execute on function public.create_chat_session(uuid, text, text, timestamptz, jsonb) to authenticated;
grant execute on function public.complete_chat_session(bigint, timestamptz) to authenticated;
grant execute on function public.create_chat_message(bigint, text, text, jsonb, integer, numeric, timestamptz) to authenticated;
grant execute on function public.create_tool_call(bigint, text, text, jsonb, jsonb, text, integer) to authenticated;

commit;
