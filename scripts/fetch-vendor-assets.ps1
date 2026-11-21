# ==============================================================================
# AeroMESH Offline UI Vendor Asset Fetcher (SHA-384 SRI Verified)
#
# Downloads the exact pinned versions of Lucide, Marked, Marked-Highlight,
# DOMPurify, Highlight.js, and KaTeX (including KaTeX .woff2 web fonts) into
# static/vendor/ so installed Windows builds run 100% offline with zero CDN
# dependency. Every top-level file is cryptographically verified against the
# exact SHA-384 Subresource Integrity (SRI) digest pinned in static/index.html.
#
# Usage:
#   pwsh -File .\scripts\fetch-vendor-assets.ps1
#   pwsh -File .\scripts\fetch-vendor-assets.ps1 -Force
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $PSScriptRoot
$vendorRoot = Join-Path $repoRoot "static\vendor"
$katexRoot = Join-Path $vendorRoot "katex"
$katexFonts = Join-Path $katexRoot "fonts"

foreach ($dir in @($vendorRoot, $katexRoot, $katexFonts)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$assets = @(
    @{
        Name   = "lucide.min.js"
        Url    = "https://cdn.jsdelivr.net/npm/lucide@0.469.0/dist/umd/lucide.min.js"
        Dest   = Join-Path $vendorRoot "lucide.min.js"
        Sha384 = "hJnF5AwidE18GSWTAGHv3ByzzvfNZ1Tcx5y1UUV3WkauuMCEzBJBMSwSt/PUPXnM"
    },
    @{
        Name   = "marked.min.js"
        Url    = "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"
        Dest   = Join-Path $vendorRoot "marked.min.js"
        Sha384 = "/TQbtLCAerC3jgaim+N78RZSDYV7ryeoBCVqTuzRrFec2akfBkHS7ACQ3PQhvMVi"
    },
    @{
        Name   = "marked-highlight.umd.js"
        Url    = "https://cdn.jsdelivr.net/npm/marked-highlight@2.2.1/lib/index.umd.js"
        Dest   = Join-Path $vendorRoot "marked-highlight.umd.js"
        Sha384 = "BjBE0bY2PIGsLgdqaC1+iwO397c7jxwJdNszWabXYxrbMXBTS9QCPxXJElH8xOE/"
    },
    @{
        Name   = "purify.min.js"
        Url    = "https://cdn.jsdelivr.net/npm/dompurify@3.2.4/dist/purify.min.js"
        Dest   = Join-Path $vendorRoot "purify.min.js"
        Sha384 = "eEu5CTj3qGvu9PdJuS+YlkNi7d2XxQROAFYOr59zgObtlcux1ae1Il3u7jvdCSWu"
    },
    @{
        Name   = "atom-one-dark.min.css"
        Url    = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css"
        Dest   = Join-Path $vendorRoot "atom-one-dark.min.css"
        Sha384 = "oaMLBGEzBOJx3UHwac0cVndtX5fxGQIfnAeFZ35RTgqPcYlbprH9o9PUV/F8Le07"
    },
    @{
        Name   = "highlight.min.js"
        Url    = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"
        Dest   = Join-Path $vendorRoot "highlight.min.js"
        Sha384 = "F/bZzf7p3Joyp5psL90p/p89AZJsndkSoGwRpXcZhleCWhd8SnRuoYo4d0yirjJp"
    },
    @{
        Name   = "katex/katex.min.css"
        Url    = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
        Dest   = Join-Path $katexRoot "katex.min.css"
        Sha384 = "n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
    },
    @{
        Name   = "katex/katex.min.js"
        Url    = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        Dest   = Join-Path $katexRoot "katex.min.js"
        Sha384 = "XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8"
    },
    @{
        Name   = "katex/auto-render.min.js"
        Url    = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        Dest   = Join-Path $katexRoot "auto-render.min.js"
        Sha384 = "+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05"
    }
)

function Get-FileSha384Base64 {
    param([Parameter(Mandatory)][string]$Path)
    $sha = [System.Security.Cryptography.SHA384]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $bytes = $sha.ComputeHash($stream)
            return [System.Convert]::ToBase64String($bytes)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha.Dispose()
    }
}

Write-Host "Fetching and verifying offline UI vendor assets into static\vendor\..." -ForegroundColor Cyan

foreach ($asset in $assets) {
    $needsDownload = $Force -or (-not (Test-Path $asset.Dest))
    if (-not $needsDownload) {
        $existingHash = Get-FileSha384Base64 -Path $asset.Dest
        if ($existingHash -eq $asset.Sha384) {
            Write-Host "  [OK] $($asset.Name) (cached, SHA-384 verified)" -ForegroundColor Green
            continue
        }
        Write-Host "  [!] $($asset.Name) hash mismatch on disk; re-downloading..." -ForegroundColor Yellow
    }

    $tmp = "$($asset.Dest).tmp"
    try {
        Invoke-WebRequest -Uri $asset.Url -OutFile $tmp -UseBasicParsing
        $actual = Get-FileSha384Base64 -Path $tmp
        if ($actual -ne $asset.Sha384) {
            Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
            throw "SHA-384 integrity mismatch for $($asset.Name)! Expected '$($asset.Sha384)', got '$actual'."
        }
        Move-Item -Path $tmp -Destination $asset.Dest -Force
        Write-Host "  [OK] $($asset.Name) (downloaded & SHA-384 verified)" -ForegroundColor Green
    } catch {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
        throw
    }
}

# Fetch KaTeX .woff2 web fonts referenced inside katex.min.css for 100% offline math rendering
$katexCssPath = Join-Path $katexRoot "katex.min.css"
$katexCss = Get-Content -Path $katexCssPath -Raw
$fontMatches = [regex]::Matches($katexCss, 'fonts/(KaTeX_[A-Za-z0-9_-]+\.woff2)')
$uniqueFonts = $fontMatches | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

Write-Host "  Ensuring $($uniqueFonts.Count) KaTeX offline .woff2 font files..." -ForegroundColor Cyan
$downloadedFonts = 0
foreach ($fontName in $uniqueFonts) {
    $fontDest = Join-Path $katexFonts $fontName
    if (-not $Force -and (Test-Path $fontDest) -and ((Get-Item $fontDest).Length -gt 0)) {
        continue
    }
    $fontUrl = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/fonts/$fontName"
    Invoke-WebRequest -Uri $fontUrl -OutFile $fontDest -UseBasicParsing
    $downloadedFonts++
}
Write-Host "  [OK] KaTeX fonts ready ($($uniqueFonts.Count) total, $downloadedFonts newly downloaded)." -ForegroundColor Green
