function F_test_TIP(data_CP_path, data_HC_path)
data_CP=load(data_CP_path);
data_HC=load(data_HC_path);
TIP_CP=data_CP.TIP;
TIP_HC=data_HC.TIP;

[h, p] = vartest2(TIP_CP,TIP_HC);
disp(p)
end


