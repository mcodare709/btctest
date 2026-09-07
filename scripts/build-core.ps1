param(
    [string]$Compiler = "g++"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $projectRoot "build"
$includeDirectory = Join-Path $projectRoot "cpp\include"
$sourceFile = Join-Path $projectRoot "cpp\src\strategy.cpp"
$testFile = Join-Path $projectRoot "cpp\tests\strategy_test.cpp"

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
& $Compiler -std=c++14 -O3 -shared $sourceFile -I $includeDirectory -o (Join-Path $buildDirectory "btc_strategy.dll")
& $Compiler -std=c++14 -O3 $testFile $sourceFile -I $includeDirectory -o (Join-Path $buildDirectory "btc_strategy_test.exe")
& (Join-Path $buildDirectory "btc_strategy_test.exe")
Write-Output "C++ strategy core built and tested: $buildDirectory"
