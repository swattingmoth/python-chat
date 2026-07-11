# Python Chat Initial Deployment
- Migrate database
- Set environment variable CHATBOT_ENV=production
- Add XAI_API_KEY to vault (service role required)
- Configure runtime secrets:
	- SUPABASE_URL
	- SUPABASE_ANON_KEY
	- SUPABASE_SERVICE_ROLE_KEY
	- IMAGE_FOLDER
- Set COOKIE_SECURE=true
- Deploy FastAPI host (`python_chat.server:app`) instead of direct Gradio launch
- Verify health endpoints: /healthz and /readyz

## Local Docker Script
- Run from project root:
	- `powershell -ExecutionPolicy Bypass -File .\deployments\deploy-local-docker.ps1`
- Optional flags:
	- `-HostPort 8080`
	- `-NoCache`
	- `-FollowLogs`
