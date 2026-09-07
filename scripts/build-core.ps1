param(
    [string]$Compiler = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot "build"
$includeDirectory = Join-Path $projectRoot "cpp\include"
$sourceFile = Join-Path $projectRoot "cpp\src\strategy.cpp"
$testFile = Join-Path $projectRoot "cpp\tests\strategy_test.cpp"
$bundledZig = Join-Path $projectRoot "tools\zig-windows-x86_64-0.13.0\zig.exe"

if (-not $Compiler) {
    $Compiler = if (Test-Path $bundledZig) { $bundledZig } else { "g++" }
}

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
if ((Split-Path -Leaf $Compiler).ToLowerInvariant() -eq "zig.exe") {
    $common = @("c++", "-std=c++14", "-O3", "-target", "x86_64-windows-gnu")
    & $Compiler @common "-shared" $sourceFile "-I" $includeDirectory "-o" (Join-Path $buildDirectory "btc_strategy.dll")
    & $Compiler @common $testFile $sourceFile "-I" $includeDirectory "-o" (Join-Path $buildDirectory "btc_strategy_test.exe")
} else {
    & $Compiler "-std=c++14" "-O3" "-shared" $sourceFile "-I" $includeDirectory "-o" (Join-Path $buildDirectory "btc_strategy.dll")
    & $Compiler "-std=c++14" "-O3" $testFile $sourceFile "-I" $includeDirectory "-o" (Join-Path $buildDirectory "btc_strategy_test.exe")
}

& (Join-Path $buildDirectory "btc_strategy_test.exe")
Write-Output "C++ strategy core built and tested: $buildDirectory"