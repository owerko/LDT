%% Wczytanie pliku
%function LDT()
clc;
clear all;
close all;
[filename,pathname] = uigetfile('*.txt','Wskaz plik danymi');
delimiterIn = '\t';
plik = importdata(fullfile(pathname,filename));
[row, col]=size(plik);
% uzytkownik wprowadza numer kolumny
fprintf('Ilosc kolumn w pliku : %1d\n',col);
kol1=input('Czas bedzie z kolumny nr :');
if isempty(kol1)
    kol1 = 1;
end
%time=plik(:,kol1);
kol2=input('Odleglosc bedzie z kolumny nr :');
if isempty(kol2)
    kol2= 1;
end

time=plik(:,kol1);
dist=plik(:,kol2);
plot(time,dist);
%% Wybor przedzialu do analizy
hold on;
n=0;
xy=[];
disp('Lewy klawiszem myszki wskazujemy punkt wykresu.')
disp('Prawy klawisz myszki wychodzi z wskazywania.')
but = 1;
while but == 1
    [xi,yi,but] = ginput(1);
    if but==1
        plot(xi,yi,'ro');
        n = n+1;
        xy(:,n) = [xi;yi];
    end
end
hold off;
close;
[rows, cols]=size(xy);

k = find(abs(time-xy(1,1)) < 0.01); % Indeks w wektorze dla czasu pierwszego klikniecia
k2 = find(abs(time-xy(1,2)) < 0.01); % jw. dla drugiego klikniecia
dtime=time(k2)-time(k); %Dlugosc analizowanego wektora w domenie czasu
ile_k=k2-k; %Ilosc probek
czestotliwosc=ile_k/dtime; %czestotliwosc probkowania

%% operacja Widmo

A=dist(k:k2);
A=A-mean(A); %normalizacja amplitudy drgan
dlugosc=length(A);
analiza=fft(A,dlugosc); %analiza=fft(dane,liczba_próbek);
widmo_am=abs(analiza);
figure(1);
plot(linspace(0,czestotliwosc,length(abs(fft(A)))),abs(fft(A)),'r'); 
%plot(linspace(start_wykresu,czestosliwosc_probkowania,dlugosc_wektora),abs(fft(wektor_danych,ilo??_probek_do_analizy)),kolor_wykresu);
Aw=A.*hamming(length(A));
figure(2);
plot(linspace(0,czestotliwosc,length(abs(fft(Aw)))),abs(fft(Aw)),'r');

f=(0:length(A)-1)'*1/czestotliwosc;
subplot(2,1,1),plot(f,A);
subplot(2,1,1),plot(f,A);

ylabel('Wykres przemieszczen'),grid on
f=(0:length(A)-1)'*czestotliwosc/length(A);
subplot(2,1,2),plot(f,abs(fft(A))); %wykres amplitudy
ylabel('Modul amplitudy'), grid on

% nfft=2^nextpow2(length(A));
% Pxx=abs(fft(A,nfft)).^2/length(A)/czestotliwosc;
% Hpsd = dspdata.psd(Pxx(1:length(Pxx)/2),'Fs',czestotliwosc); %PSD
% figure(3);
% plot(Hpsd);

%% Znajdowanie [automatyczne] czestotliwosci
minPeakHeight = 3*std(abs(fft(A)));
[pks, locs] = findpeaks(abs(fft(A)), 'MINPEAKHEIGHT', minPeakHeight,'SortStr','descend');

% Ilosc wykrytych postaci drgan
ile_postaci = numel(pks);

hold on;
plot(f(locs), pks, 'r', 'Marker', 'v', 'LineStyle', 'none');
%xlabel('Czas(s)');
%ylabel('Amplituda');
hold off;

% Indeksy znalezionych czestotliwosci
%F_znalezione=f(locs);
%F_ok=F_znalezione(F_znalezione<(czestotliwosc/2)); %Wybieramy te pod pod f/2
%[FileName,PathName] = uiputfile
%name=fullfile(PathName,FileName);
%dlmwrite(name,F_ok); % Zapis do pliku - format CSV
%% Filtrowanie
% Projekt filtra o oknie 0.4 Hz
B=fir1(200,(f(locs(1))-0.2)/czestotliwosc,'high'); %projekt filtra
  A=filtfilt(B,1,A);
B=fir1(200,(f(locs(1))+0.2)/czestotliwosc*2); %projekt filtra
  A=filtfilt(B,1,A); 
figure(3);
t=time(k:k2);
plot(t,A);

%% Transformacja Hilberta
s_hilbert=hilbert(A);
obwiednia=(abs(s_hilbert));
obwiednia=obwiednia(0.05*length(obwiednia):0.95*length(obwiednia));
figure(4);
hold on;
plot (t,A);
title('analiza Hilberta');
ylabel('przemieszczenie (mm), amplituda');
xlabel('czas pomiaru (s)');
hold on;
t=time(k:k2);
%obwiednia=obwiednia(0.02*length(t):0.98*length(t));
t=t(0.05*length(t):  0.95*length(t));
plot(t,obwiednia,'r');
%ss = fitoptions('Method','NonlinearLeastSquares','Bisquare');
%ff = fittype('a*exp(-x*b)','options',ss); 
%cfun2 = fit(t,obwiednia,ff);
%cfun2
%plot(cfun2,'g');

%modelfun=@(a,x)a(1)*exp(-x*a(2));
%beta0=[0.5 0.5];
%mdl=fitnlm(t,obwiednia,modelfun,beta0);
%plot(mdl);
%mdl
%xlabel('czas (s)');
%ylabel('amplituda drgan skladowej fs');
%legend({'sygnal','obwiednia tr Hilberta','a*exp(-x*b)'});
%hold off;

figure(7);
hold off;
plot(t,log(obwiednia));
hold on;
ss = fitoptions('Method','NonlinearLeastSquares');
ff = fittype('a*x+b','options',ss);
cfun2 = fit(t,log(obwiednia),ff);
cfun2
plot(cfun2,'r');
xlabel('czas (s)');
ylabel('log-amplituda drgan skladowej fs');
legend({'log-obwiednia tr Hilberta','ax+b'});



