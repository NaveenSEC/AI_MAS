## Development Hints

### 1. Clone the repository
```
git clone https://github.com/NaveenSEC/AI_MAS.git
```

### 2. Enter the project folder
```
cd AI_MAS
```

### 3. Fetch all remote branches and switch to the specific branch
```
git fetch origin
git checkout claude/optimistic-planck-fag80e
```

### 4. Open the Project in VS Code

```
code .
```

(Edit Your Code inside VS Code)
### 6. Check which files were modified

```
git status
```

### 7. Stage your changes
```
git add .
```

### 8. Commit your changes with a descriptive message
```
git commit -m "Update code with my edits"
```

### 9. Push the changes back to the exact same branch on GitHub
```
git push origin claude/optimistic-planck-fag80e
```

## Resources

```
1. Introduction to agent skills - https://anthropic.skilljar.com/introduction-to-agent-skills
2. TODO
```

## General 
### How to update the Tokens
Generate the Tokens in Github: Settings -> Developer Settings -> Personal access tokens -> Tokens (classic) -> Generate new token (classic)

```
git remote set-url origin https://NaveenSEC:YOUR_TOKEN@github.com/NaveenSEC/Cryptography.git
```

### How to check SSH connection
```
ssh -T git@github.com
```
You should be seeing - Hi NaveenSEC! You've successfully authenticated, but GitHub does not provide shell access.

## Trouble Shooting
When HTTPS token authentication continues returning a 403 error despite updating the token, it usually means Git is overriding your token with cached credentials stored in the Windows Credential Manager or Git Credential Manager, or the token missing explicit write scope for that specific repository.

Switching to SSH bypasses the HTTPS credential caching issue entirely and gives you a reliable, permanent setup.

Solution 1: Switch to SSH (Fastest & Guaranteed Fix)
Since you already verified your SSH host key earlier, switch the remote URL to SSH:

### Set the remote URL to SSH:

```
git remote set-url origin git@github.com:NaveenSEC/Cryptography.git
```

### Test your SSH connection:

```
ssh -T git@github.com
```

### Push to GitHub:

```
git push -u origin main
```

## Author
Naveenkumar Balasubramanian \
Cybersecurity Researcher
