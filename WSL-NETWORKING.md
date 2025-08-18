# WSL Networking Setup for Permanent Access

This document explains how to permanently access your FastAPI app running in WSL from Windows localhost.

## Quick Setup

1. **Run the setup script as Administrator:**
   ```cmd
   # Right-click "setup-wsl-networking.bat" and select "Run as administrator"
   ```

2. **If WSL gets a new IP address later, just run:**
   ```powershell
   # Right-click PowerShell and "Run as administrator"
   .\setup-port-forwarding.ps1
   ```

## What the Setup Does

### 1. WSL Configuration (`.wslconfig`)
- Enables `localhostForwarding=true` for automatic port forwarding
- Sets up NAT networking mode for better connectivity
- Copied to `%USERPROFILE%\.wslconfig`

### 2. Port Forwarding Rules
- Creates Windows netsh rule: `0.0.0.0:8000 -> WSL_IP:8000`
- Automatically detects current WSL IP address
- Updates the rule when WSL IP changes

### 3. Windows Firewall
- Creates inbound rule to allow traffic on port 8000
- Named "WSL Port 8000" for easy identification

## Manual Commands (if needed)

### Check current WSL IP:
```bash
wsl hostname -I
```

### Check port forwarding rules:
```cmd
netsh interface portproxy show all
```

### Manually add port forwarding (as Admin):
```cmd
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=WSL_IP
```

### Remove port forwarding rule:
```cmd
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0
```

## Troubleshooting

### If localhost:8000 still doesn't work:
1. Restart WSL: `wsl --shutdown` then `wsl`
2. Run setup script again as Administrator
3. Check Windows Firewall settings
4. Verify app is bound to `0.0.0.0:8000` (not `127.0.0.1:8000`)

### Alternative Access Methods:
- Direct WSL IP: `http://WSL_IP:8000`
- Get WSL IP: `wsl hostname -I`

## App Configuration

Your app is already configured correctly in `run.sh`:
```bash
uvicorn app:app --reload --port 8000 --host 0.0.0.0
```

The `--host 0.0.0.0` ensures the app accepts connections from any interface.