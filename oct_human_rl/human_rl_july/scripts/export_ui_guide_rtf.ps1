param(
    [string]$InputPath = "docs/UI_GUIDE.md",
    [string]$OutputPath = "docs/UI_GUIDE.rtf"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

function Escape-Rtf([string]$Text) {
    $result = [System.Text.StringBuilder]::new()
    foreach ($char in $Text.ToCharArray()) {
        $number = [int][char]$char
        if ($char -eq '\') { [void]$result.Append('\\') }
        elseif ($char -eq '{') { [void]$result.Append('\{') }
        elseif ($char -eq '}') { [void]$result.Append('\}') }
        elseif ($number -ge 32 -and $number -le 126) { [void]$result.Append($char) }
        elseif ($number -notin 10, 13) {
            if ($number -gt 32767) { $number -= 65536 }
            [void]$result.Append("\u$number?")
        }
    }
    $result.ToString()
}

function Inline-Rtf([string]$Text) {
    $Text = [regex]::Replace($Text, '\[([^\]]+)\]\(([^)]+)\)', '$1 ($2)')
    $Text = $Text.Replace('**', '').Replace('`', '')
    Escape-Rtf $Text
}

function Image-Rtf([string]$Path, [string]$AltText) {
    $image = [System.Drawing.Image]::FromFile($Path)
    try {
        $width = $image.Width
        $height = $image.Height
    }
    finally { $image.Dispose() }

    $displayWidth = if ($height -gt $width) { 4800 } else { 10800 }
    $displayHeight = [math]::Round($displayWidth * $height / $width)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $hex = [System.BitConverter]::ToString($bytes).Replace('-', '')
    $hex = [regex]::Replace($hex, '(.{128})', "`$1`n")
    $alt = Escape-Rtf $AltText
    "\pard\qc\sa120{\*\picprop{\sp{\sn wzDescription}{\sv $alt}}}{\pict\pngblip\picw$width\pich$height\picwgoal$displayWidth\pichgoal$displayHeight`n$hex}\par`n"
}

$source = (Resolve-Path -LiteralPath $InputPath).Path
$sourceDirectory = Split-Path -Parent $source
$body = [System.Text.StringBuilder]::new()
$inCode = $false
$tableHeader = $false

foreach ($line in Get-Content -LiteralPath $source -Encoding UTF8) {
    if ($line -match '^```') {
        $inCode = -not $inCode
        if (-not $inCode) { [void]$body.Append("\pard\sa100\par`n") }
        continue
    }

    if ($inCode) {
        [void]$body.Append("\pard\li360\ri360\f1\fs18\highlight8 $(Escape-Rtf $line)\line`n")
        continue
    }

    if ($line -match '^!\[([^\]]*)\]\(([^)]+)\)$') {
        $imagePath = Join-Path $sourceDirectory $Matches[2]
        [void]$body.Append((Image-Rtf $imagePath $Matches[1]))
    }
    elseif ($line -match '^# (.+)$') {
        [void]$body.Append("\pard\keepn\sb120\sa200\b\fs38\cf1 $(Inline-Rtf $Matches[1])\b0\par`n")
    }
    elseif ($line -match '^## (.+)$') {
        [void]$body.Append("\pard\keepn\sb280\sa130\brdrb\brdrs\brdrw12\brdrcf2\b\fs28\cf1 $(Inline-Rtf $Matches[1])\b0\par`n")
    }
    elseif ($line -match '^### (.+)$') {
        [void]$body.Append("\pard\keepn\sb200\sa90\b\fs23\cf2 $(Inline-Rtf $Matches[1])\b0\par`n")
    }
    elseif ($line -match '^\|(.+)\|$') {
        $cells = @($Matches[1].Split('|') | ForEach-Object { $_.Trim() })
        $divider = ($cells | Where-Object { $_ -notmatch '^:?-{3,}:?$' }).Count -eq 0
        if (-not $divider) {
            $content = ($cells | ForEach-Object { Inline-Rtf $_ }) -join '\tab '
            $weight = if (-not $tableHeader) { '\b ' } else { '' }
            $weightEnd = if (-not $tableHeader) { '\b0 ' } else { '' }
            [void]$body.Append("\pard\tx2600\tx5300\tx7900\tx10500\sb65\sa65\fs18 $weight$content$weightEnd\par`n")
            $tableHeader = $true
        }
    }
    elseif ($line -match '^-\s+(.+)$') {
        $tableHeader = $false
        [void]$body.Append("\pard\li620\fi-300\tx620\sa70\fs20\bullet\tab $(Inline-Rtf $Matches[1])\par`n")
    }
    elseif ($line -match '^\d+\.\s+(.+)$') {
        $tableHeader = $false
        $number = $line.Substring(0, $line.IndexOf('.') + 1)
        [void]$body.Append("\pard\li620\fi-340\tx620\sa70\fs20 $number\tab $(Inline-Rtf $Matches[1])\par`n")
    }
    elseif ($line -match '^\*(.+)\*$') {
        $tableHeader = $false
        [void]$body.Append("\pard\qc\sa150\i\fs18\cf6 $(Inline-Rtf $Matches[1])\i0\par`n")
    }
    elseif ([string]::IsNullOrWhiteSpace($line)) {
        $tableHeader = $false
        [void]$body.Append("\pard\sa35\par`n")
    }
    else {
        $tableHeader = $false
        [void]$body.Append("\pard\sa120\sl276\slmult1\fs20\cf1 $(Inline-Rtf $line)\par`n")
    }
}

$header = "{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0 Segoe UI;}{\f1 Consolas;}}{\colortbl;\red18\green43\blue52;\red15\green118\blue110;\red79\green70\blue229;\red180\green95\blue22;\red47\green122\blue61;\red100\green116\blue139;\red207\green216\blue220;\red236\green246\blue245;}\paperw12240\paperh15840\margl720\margr720\margt720\margb720\widowctrl\viewkind4\uc1\f0\fs20`n"
$target = [System.IO.Path]::GetFullPath($OutputPath)
[System.IO.File]::WriteAllText($target, $header + $body.ToString() + "}`n", [System.Text.Encoding]::ASCII)
$file = Get-Item -LiteralPath $target
Write-Output "Created $($file.FullName) ($($file.Length) bytes)"
