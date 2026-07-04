-- Harden vault secret access.
-- Restrict secret reads to service_role callers only and lock down function execute privileges.

begin;

create or replace function public.get_from_vault(
    p_secret_name text
)
returns text
language plpgsql
security definer
set search_path = pg_catalog, public, auth, vault
as $$
declare
    v_secret text;
begin
    if auth.role() <> 'service_role' then
        raise exception 'get_from_vault is restricted to service_role'
            using errcode = '42501';
    end if;

    select ds.decrypted_secret
    into v_secret
    from vault.decrypted_secrets as ds
    where ds.name = p_secret_name
    limit 1;

    return v_secret;
end;
$$;

revoke all on function public.get_from_vault(text) from public;
revoke all on function public.get_from_vault(text) from anon;
revoke all on function public.get_from_vault(text) from authenticated;
grant execute on function public.get_from_vault(text) to service_role;

commit;
