[CmdletBinding()]
param (
    [string[]]
    $File,

    [switch]
    $ExportAsObject
)

function Convert-CurrencyStringToDecimal ([string]$input)
{
    ((($input -replace '\$') -replace '[)]') -replace '\(', '-') -replace '[^-0-9.]'
}

$global:outData = [System.Collections.ArrayList]::new()

foreach ($f in $File)
{
    if (-not (Test-Path $f))
    {
        throw "Cannot open file '$f'."
    }

    # read csv file
    $content = Get-Content -Path $f
    # find the lines that contain price information
    $csvdata = $content | ? {$_ -match ";.*;"} | ConvertFrom-Csv -Delimiter ';'

    # filter just the lines with (OHLC on them and make into CSV structure
    $data = $csvData | ? {$_ -match "\(OHLC"}

    foreach ($item in $data)
    {
        # capture the OHLC data
        $null = $item.Strategy -match "\(OHLC\|(.*)\)"
        $v = $Matches[1] -split '\|'

        $open  = $v[0] | Convert-CurrencyStringToDecimal
        $high  = $v[1] | Convert-CurrencyStringToDecimal
        $low   = $v[2] | Convert-CurrencyStringToDecimal
        $close = $v[3] | Convert-CurrencyStringToDecimal

        # add to our $outData array
        $null = $outData.Add(
            [PSCustomObject]@{
                'DateTime' = ([datetime]::Parse($item.'Date/Time'))
                'Open' = [decimal]$open
                'High' = [decimal]$high
                'Low' = [decimal]$low
                'Close' = [decimal]$close
                }
            )
    }
}

if ($ExportAsObject)
{
    # helpful message to show caller our output variable
    Write-Output "Out Data $($outData.Count) items (exported as `$outData)"
}
else
{
    # don't show any output, and just return the data to the pipeline
    return $outData
}