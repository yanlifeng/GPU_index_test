grep btid $1 | awk '
{
    for (i = 3; i <= NF; i++) {
        if ($i ~ /^[0-9.]+$/)  # 只处理数值
            sum[i] += $i;
    }
}
END {
    for (i = 3; i <= NF; i++) {
        if (sum[i] != "")
            printf "%s ", sum[i];
    }
    print "";
}'

