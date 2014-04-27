load '../data/EEG_EYE.txt';
X = EEG_EYE(:,1:14)';
y = EEG_EYE(:,15)';

net = newff(X,y,[6])
[net, trainer,out,error] = train(net, X, y);