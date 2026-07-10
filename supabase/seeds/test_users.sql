-- Optional local test users for development.
-- Requires Supabase local auth service and may be environment-specific.
-- Keep this file as a reference seed; adjust IDs/password hashes per environment.

-- Intentionally left minimal because auth.users inserts typically require
-- managed flows through GoTrue and are not always safe to seed directly.
-- Enable pgcrypto for password hashing
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Generate a random UUID for the user
DO $$
DECLARE
  v_user_id UUID :='c1c2e36c-209f-49fb-aea9-0536c0c30263';
  -- The password is 'password123'
  v_encrypted_pw TEXT := crypt('password123', gen_salt('bf'));
BEGIN

  -- 1. Insert the user into auth.users
  INSERT INTO auth.users (
    id,
    instance_id,
    aud,
    role,
    email,
    encrypted_password,
    email_confirmed_at,
    raw_app_meta_data,
    raw_user_meta_data,
    confirmation_token,
    recovery_token,
    email_change_token_new,
    email_change,
    created_at,
    updated_at
  )
  VALUES (
    v_user_id,
    '00000000-0000-0000-0000-000000000000',
    'authenticated',
    'authenticated',
    'user@example.com',
    v_encrypted_pw,
    NOW(),
    '{"provider":"email","providers":["email"]}',
    '{"first_name": "Demo", "last_name": "User"}',
    '',
    '',
    '',
    '',
    NOW(),
    NOW()
  );

  -- 2. Link an identity so the user can actually log in
  INSERT INTO auth.identities (
    id,
    user_id,
    identity_data,
    provider,
    provider_id,
    last_sign_in_at,
    created_at,
    updated_at
  )
  VALUES (
    v_user_id,
    v_user_id,
    format('{"sub": "%s", "email": "user@example.com"}', v_user_id)::jsonb,
    'email',
    v_user_id,
    NOW(),
    NOW(),
    NOW()
  );

END $$;