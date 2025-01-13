grep "btid" log | awk '{t1+=$6; t2+=$7; t3+=$8; t4+=$11; t5+=$12; t6+=$13; t7+=$14; t8+=$15; t9+=$16; t10+=$17} END {print t1, t2, t3, t4, t5, t6, t7, t8, t9, t10}'
