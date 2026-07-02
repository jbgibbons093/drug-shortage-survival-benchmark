# Compatibility wrapper for the canonical Python orchestrator.
#
# The old version of this file maintained a separate stage list and drifted
# into calling archived analyses. Keep this wrapper only so existing shell
# habits still work. The active stage definition now lives in 28_pipeline.py.

param(
  [switch]$FromModels,
  [switch]$PredictionsOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$argsForPipeline = @("Programs\28_pipeline.py")
if ($FromModels -or $PredictionsOnly) {
  $argsForPipeline += "--from-models"
}

& python @argsForPipeline
exit $LASTEXITCODE
