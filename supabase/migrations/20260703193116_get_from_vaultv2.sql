create or replace function public.get_from_vault(
    p_secret_name text
)
returns text
language plpgsql
security invoker
set search_path = public
as $$
begin
    return (
        select decrypted_secret as secret
        from vault.decrypted_secrets
        where name = p_secret_name
    );
end;
$$;