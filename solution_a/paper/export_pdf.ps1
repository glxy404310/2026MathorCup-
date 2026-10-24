param(
    [string]$InputDocx = (Join-Path $PSScriptRoot '啤酒风味建模提交稿.docx'),
    [string]$OutputPdf = (Join-Path $PSScriptRoot 'qa/main.pdf')
)
$ErrorActionPreference = 'Stop'
$inputPath = (Resolve-Path -LiteralPath $InputDocx).Path
$outputPath = [IO.Path]::GetFullPath($OutputPdf)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($inputPath, $false, $true)
    $document.Repaginate()
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $document.Fields.Update() | Out-Null
    $document.ExportAsFixedFormat($outputPath, 17)
    if (-not (Test-Path -LiteralPath $outputPath)) { throw 'PDF export did not create a file.' }
    Get-Item -LiteralPath $outputPath | Select-Object FullName, Length
} finally {
    if ($null -ne $document) { try { $document.Close(0) } catch { Write-Verbose $_ } }
    if ($null -ne $word) { try { $word.Quit() } catch { Write-Verbose $_ } }
}
