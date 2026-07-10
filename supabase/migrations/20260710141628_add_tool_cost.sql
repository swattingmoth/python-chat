alter table public.tool_calls add column estimated_cost numeric(10, 2) default 0.0;

CREATE OR REPLACE FUNCTION public.create_tool_call(
	p_message_id bigint,
	p_tool_name text,
	p_status text,
	p_input_args jsonb DEFAULT NULL::jsonb,
	p_output_result jsonb DEFAULT NULL::jsonb,
	p_error_message text DEFAULT NULL::text,
	p_latency_ms integer DEFAULT NULL::integer,
	p_estimated_cost numeric(10, 2) DEFAULT 0.0
    )
    RETURNS bigint
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
declare
    new_id bigint;
begin
    insert into public.tool_calls (
        message_id,
        tool_name,
        status,
        input_args,
        output_result,
        error_message,
        latency_ms,
        estimated_cost
    )
    values (
        p_message_id,
        p_tool_name,
        p_status,
        p_input_args,
        p_output_result,
        p_error_message,
        p_latency_ms,
        p_estimated_cost
    )
    returning id into new_id;

    return new_id;
end;
$BODY$;