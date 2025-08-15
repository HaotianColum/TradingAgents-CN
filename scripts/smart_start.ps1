# TradingAgents-CN 智能Docker启动脚本 (Windows PowerShell版本)
# 功能：自动判断是否需要重新构建Docker镜像
# 使用：powershell -ExecutionPolicy Bypass -File scripts\smart_start.ps1
# 
# 判断逻辑：
# 1. 检查是否存在tradingagents-cn镜像
# 2. 如果镜像不存在 -> 执行构建启动
# 3. 如果镜像存在但代码有变化 -> 执行构建启动  
# 4. 如果镜像存在且代码无变化 -> 快速启动

Write-Host "=== TradingAgents-CN Docker Smart Start ===" -ForegroundColor Green
Write-Host "Env: Windows PowerShell" -ForegroundColor Cyan

# 检查是否有镜像
$imageExists = docker images | Select-String "tradingagents-cn"

if ($imageExists) {
    Write-Host "Found existing image" -ForegroundColor Green

    # Check if code changed (simple)
    $gitStatus = git status --porcelain
    if ([string]::IsNullOrEmpty($gitStatus)) {
        Write-Host "No code changes, fast start" -ForegroundColor Blue
        docker-compose up -d
    } else {
        Write-Host "Changes detected, rebuilding" -ForegroundColor Yellow
        docker-compose up -d --build
    }
} else {
    Write-Host "First run, building image" -ForegroundColor Yellow
    docker-compose up -d --build
}

Write-Host "Start complete" -ForegroundColor Green
Write-Host "Web: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Redis Admin: http://localhost:8081" -ForegroundColor Cyan