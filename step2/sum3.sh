grep time3 "$1" | awk '
{
    sum1 += $4;
    sum2 += $6;
    sum3 += $7;
    sum4 += $9;
    sum5 += $10;
    sum6 += $12;
    sum7 += $13;
    sum8 += $15;
    sum9 += $16;
    sum10 += $17;
    sum11 += $19;
    sum12 += $20;
    sum13 += $21;
    sum14 += $22;
    sum15 += $23;
    sum16 += $25;
}
END {
    printf "Sum: %.3f, %.3f %.3f ( %.3f %.3f ) %.3f %.3f ( %.3f %.3f %.3f [ %.3f %.3f %.3f %.3f %.3f ]) %.3f\n",
        sum1,          sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sum16, sum17;
}' 

