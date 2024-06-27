function [SC] = Filter_SC(SC)

fc = 2000; %sampling frequency
fn = fc/2; %Nyquist frequency
ft = 2; %Cutting frequency at 2 Hz
[b,a] = cheby1(3,0.5,ft/fn);

SC = filtfilt(b,a, SC);
end