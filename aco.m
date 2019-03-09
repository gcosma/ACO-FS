% Gnetic algorithm For feature selection by Sadegh Salesi and Georgina Cosma      %
% Programmed by Sadegh Salesi at Nottignham Trent University              %
% Last revised:  2017     %
% Reference: S. Salesi and G. Cosma, A novel extended binary cuckoo search algorithm for feature selection, 2017 2nd International Conference on Knowledge Engineering and Applications (ICKEA), London, 2017, pp. 6-12.
% https://ieeexplore.ieee.org/document/8169893
% Copyright (c) 2017, Sadegh Salesi and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------

clc;
clear;
close all;
for nrun=1:10

%% Problem Definition

% model=CreateModel();
% 
% CostFunction=@(x) MyCost(x,model);
% 
% nVar=model.n;


X=xlsread('data_ovr');
Y=xlsread('target_ovr');
nVar=size(X,2);


%% ACO Parameters
fen=10000;


nAnt=30;        % Number of Ants (Population Size)

MaxIt=floor(fen/nAnt);      % Maximum Number of Iterations
% MaxIt=20;

Q=1;

tau0=0.1;        % Initial Phromone

alpha=1;        % Phromone Exponential Weight
% beta=0.02;      % Heuristic Exponential Weight

rho=0.1;        % Evaporation Rate


%% Initialization

N=[0 1];

% eta=[model.w./model.v
%      model.v./model.w];           % Heuristic Information

tau=tau0*ones(2,nVar);      % Phromone Matrix

BestCost=zeros(MaxIt,1);    % Array to Hold Best Cost Values

% Empty Ant
empty_ant.Tour=[];
empty_ant.x=[];
empty_ant.Cost=[];
empty_ant.Sol=[];
empty_ant.acc=[];
empty_ant.nfeat=[];
empty_ant.t=[];

% Ant Colony Matrix
ant=repmat(empty_ant,nAnt,1);

% Best Ant
BestSol.Cost=inf;
nfeat=0;
acc=0;


%% ACO Main Loop
tic
for it=1:MaxIt
    % Move Ants
    for k=1:nAnt
        
        ant(k).Tour=[];
        
        for l=1:nVar
            
            j=round(1+rand(1));
%             
%             P=tau(:,l).^alpha.*eta(:,l).^beta;
%             
%             P=P/sum(P);
%             
%             j=RouletteWheelSelection(P);
            
            
            ant(k).Tour=[ant(k).Tour j];
            
        end
        
        ant(k).x=N(ant(k).Tour);
        if sum(ant(k).x)>0
             [ant(k).Cost,ant(k).acc,ant(k).nfeat]=svm(X,Y,ant(k).x);
        else
            ant(k).Cost=Inf;
        end
        
        if ant(k).Cost<BestSol.Cost
            BestSol=ant(k);
            nfeat=ant(k).nfeat;
            acc=ant(k).acc;
        end
        
    end
    
    % Update Phromones
    for k=1:nAnt
        
        tour=ant(k).Tour;
        
        for l=1:nVar
            
            tau(tour(l),l)=tau(tour(l),l)+Q/ant(k).Cost;
            
        end
        
    end
    
    % Evaporation
    tau=(1-rho)*tau;
    
    % Store Best Cost
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
%     if BestSol.Sol.IsFeasible
%         FeasiblityFlag = '*';
%     else
%         FeasiblityFlag = '';
%     end
    disp([' Run = ' num2str(nrun)  ' Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))  ' Acc = ' num2str(acc) ' Nfeat = ' num2str(nfeat)]);
    
end
save(nrun,1)=acc;
save(nrun,2)=nfeat;
save(nrun,3)=toc;
end

%% Results

% figure;
% plot(BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;
