%% Developed by Amin Nazari 
% 	aminnazari91@gmail.com 
%	0918 546 2272

clc;
clear;
close all;
warning off all;

maxTrainingTime = 24.0 * 3600; 
startTime = tic;

while toc(startTime) < maxTrainingTime
    %% Create sensor nodes, Set Parameters and Create Energy Model 
    %%%%%%%%%%%%%%%%%%%%%%%%% Initial Parameters %%%%%%%%%%%%%%%%%%%%%%%
    n=100;                                  %Number of Nodes in the field
    [Area,Model]=setParameters(n);     		%Set Parameters Sensors and Network
    
    %%%%%%%%%%%%%%%%%%%%%%%%% configuration Sensors %%%%%%%%%%%%%%%%%%%%
    CreateRandomSen(Model,Area);            %Create a random scenario
    load Locations                          %Load sensor Location
    Sensors=ConfigureSensors(Model,n,X,Y);
    ploter(Sensors,Model);                  %Plot sensors
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Parameters initialization %%%%%%%%%%%%%%%%
    countCHs=0;         %counter for CHs
    flag_first_dead=0;  %flag_first_dead
    deadNum=0;          %Number of dead nodes
    
    initEnergy=0;       %Initial Energy
    for i=1:n
          initEnergy=Sensors(i).E+initEnergy;
    end
    
    SRP=zeros(1,Model.rmax);    %number of sent routing packets
    RRP=zeros(1,Model.rmax);    %number of receive routing packets
    SDP=zeros(1,Model.rmax);    %number of sent data packets 
    RDP=zeros(1,Model.rmax);    %number of receive data packets 
    
    Sum_DEAD=zeros(1,Model.rmax);
    CLUSTERHS=zeros(1,Model.rmax);
    AllSensorEnergy=zeros(1,Model.rmax);
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Build GCN Model  %%%%%%%%%%%%%%%%%%%%%%%%%
    pe = pyenv;
        if ~strcmp(pe.Status, 'Loaded')
            error('Python environment is not loaded. Bitte konfiguriere pyenv.');
        end
    
    A = createAdjacencyMatrix(Sensors, Model);        
    A_batch = reshape(A, [1, size(A,1), size(A,2)]);
    A_py = py.numpy.array(single(A_batch));
    
    optimizer = py.tensorflow.keras.optimizers.Adam(0.0005);
    num_nodes = Model.n;
    feature_dim = 6;
    
    gcn_module = py.importlib.import_module('gcn_model');    
    
    model_file = 'trained_gcn_model.keras';
    
    if isfile(model_file)
        custom_objects = py.dict(pyargs(...
        'GraphConvolution', gcn_module.GraphConvolution, ...
        'normalize_adjacency', gcn_module.normalize_adjacency));
        
        gcnModel = py.tensorflow.keras.models.load_model(model_file, ...
                   pyargs('safe_mode',false,'custom_objects',custom_objects));
    
        fprintf('Modell geladen: %s\n', model_file);
    else
        gcn_module = py.importlib.import_module('gcn_model');        
        hidden_units = int32(256);
        gcnModel = gcn_module.build_gcn_model(int32(num_nodes), int32(feature_dim), hidden_units);
        fprintf('Kein gespeichertes Modell gefunden, neues Modell erstellt.\n');
    end
    
    prevClusterHeads = [];          
    % Initialisierung dynamischer Penalty-Werte und Inkrement
    repeated_penalty_dyn = 0.5;
    proximity_penalty_dyn = 0.5;
    ch_ratio_penalty_dyn  = 0.5;
    penalty_increment = 0.5;
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Start Simulation %%%%%%%%%%%%%%%%%%%%%%%%%
    global srp rrp sdp rdp
    srp=0;          %counter number of sent routing packets
    rrp=0;          %counter number of receive routing packets
    sdp=0;          %counter number of sent data packets 
    rdp=0;          %counter number of receive data packets 
    
    %Sink broadcast start message to all nodes
    Sender=n+1;     %Sink
    Receiver=1:n;   %All nodes
    Sensors=SendReceivePackets(Sensors,Model,Sender,'Hello',Receiver);
    
    % All sensor send location information to Sink .
     Sensors=disToSink(Sensors,Model);
    % Sender=1:n;     %All nodes
    % Receiver=n+1;   %Sink
    % Sensors=SendReceivePackets(Sensors,Model,Sender,'Hello',Receiver);
    
    %Save metrics
    SRP(1)=srp;
    RRP(1)=rrp;  
    SDP(1)=sdp;
    RDP(1)=rdp;
    
    %% Main loop program
    for r=1:1:Model.rmax
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%
        %This section Operate for each epoch   
        member=[];              %Member of each cluster in per period
        countCHs=0;             %Number of CH in per period
        %counter for bit transmitted to Bases Station and Cluster Heads
        srp=0;          %counter number of sent routing packets
        rrp=0;          %counter number of receive routing packets
        sdp=0;          %counter number of sent data packets to sink
        rdp=0;          %counter number of receive data packets by sink
        %initialization per round
        SRP(r+1)=srp;
        RRP(r+1)=rrp;  
        SDP(r+1)=sdp;
        RDP(r+1)=rdp;   
        pause(0.001)    %pause simulation
        hold off;       %clear figure
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Sensors=resetSensors(Sensors,Model);
        %allow to sensor to become cluster-head. LEACH Algorithm  
        %AroundClear=10;
        %if(mod(r,AroundClear)==0) 
        %    for i=1:1:n
        %        Sensors(i).G=0;
        %    end
        %end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% plot sensors %%%%%%%%%%%%%%%%%%%%%%%
        deadNum=ploter(Sensors,Model);
        
        %Save r'th period When the first node dies
        if (deadNum>=1)      
            if(flag_first_dead==0)
                first_dead=r;
                flag_first_dead=1;
            end  
        end
        
    %%%%%%%%%%%%%%%%%%%%%%% cluster head election %%%%%%%%%%%%%%%%%%%
        
        X = createFeatureMatrix(Sensors, Model);      % X:[Model.n x 6]
        probs = runGCNModel(X,A_py,gcnModel);
    
        [TotalCH,Sensors]=filterCH(Sensors,Model,r,probs); 
        
        %Broadcasting CHs to All Sensor that are in Radio Rage CH.
        for i=1:length(TotalCH)
            
            Sender=TotalCH(i).id;
            SenderRR=Model.RR;
            Receiver=findReceiver(Sensors,Model,Sender,SenderRR);   
            Sensors=SendReceivePackets(Sensors,Model,Sender,'Hello',Receiver);
                
        end 
        
        %Sensors join to nearest CH 
        Sensors=JoinToNearestCH(Sensors,Model,TotalCH);
        
    %%%%%%%%%%%%%%%%%%%%%%% end of cluster head election phase %%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%% plot network status in end of set-up phase 
    
        for i=1:n
            
            if (Sensors(i).type=='N' && Sensors(i).dis2ch<Sensors(i).dis2sink && ...
                    Sensors(i).E>0)
                
                XL=[Sensors(i).xd ,Sensors(Sensors(i).MCH).xd];
                YL=[Sensors(i).yd ,Sensors(Sensors(i).MCH).yd];
                hold on
                line(XL,YL)
                
            end
            
        end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% steady-state phase %%%%%%%%%%%%%%%%%
        NumPacket=Model.NumPacket;
        for i=1:1:1%NumPacket 
            
            %Plotter     
            deadNum=ploter(Sensors,Model);
            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% All sensor send data packet to  CH 
            for j=1:length(TotalCH)
                
                Receiver=TotalCH(j).id;
                Sender=findSender(Sensors,Model,Receiver); 
                Sensors=SendReceivePackets(Sensors,Model,Sender,'Data',Receiver);
                
            end
            
        end
        
        
    %%%%%%%%%%%% send Data packet from CH to Sink after Data aggregation
        for i=1:length(TotalCH)
                
            Receiver=n+1;               %Sink
            Sender=TotalCH(i).id;       %CH 
            Sensors=SendReceivePackets(Sensors,Model,Sender,'Data',Receiver);
                
        end
    %%% send data packet directly from other nodes(that aren't in each cluster) to Sink
        for i=1:n
            if(Sensors(i).MCH==Sensors(n+1).id)
                Receiver=n+1;               %Sink
                Sender=Sensors(i).id;       %Other Nodes 
                Sensors=SendReceivePackets(Sensors,Model,Sender,'Data',Receiver);
            end
        end
     
       
    %% STATISTICS
         
        Sum_DEAD(r+1)=deadNum;
        
        SRP(r+1)=srp;
        RRP(r+1)=rrp;  
        SDP(r+1)=sdp;
        RDP(r+1)=rdp;
        
        CLUSTERHS(r+1)=countCHs;
        
        alive=0;
        SensorEnergy=0;
        for i=1:n
            if Sensors(i).E>0
                alive=alive+1;
                SensorEnergy=SensorEnergy+Sensors(i).E;
            end
        end
        AliveSensors(r)=alive; %#ok
        
        SumEnergyAllSensor(r+1)=SensorEnergy; %#ok
        
        AvgEnergyAllSensor(r+1)=SensorEnergy/alive; %#ok
        
        ConsumEnergy(r+1)=(initEnergy-SumEnergyAllSensor(r+1))/n; %#ok
    
        RoundEnergy(r+1)=SumEnergyAllSensor(r) - SumEnergyAllSensor(r+1);
        
        
        %%%%%%%%%%%%%%%%%%%% RL Training %%%%%%%%%%%%%%%%%%%%%%
        
        X_batch = reshape(X, [1, num_nodes, feature_dim]);
        
        X_py = py.numpy.array(X_batch);
    
        if ~isempty(TotalCH)
            currentCH = sort([TotalCH.id]);
        else
            currentCH = [];
        end    
       
        if (length(TotalCH) < 0.05 * alive)
            ch_ratio_penalty_dyn = ch_ratio_penalty_dyn + penalty_increment;
        else
            ch_ratio_penalty_dyn = max(ch_ratio_penalty_dyn - penalty_increment, 0);
         
            if isequal(prevClusterHeads, currentCH)
                repeated_penalty_dyn = repeated_penalty_dyn + penalty_increment;
            else 
                repeated_penalty_dyn = max(repeated_penalty_dyn - penalty_increment, 0);
            end
        end
        
        prevClusterHeads = currentCH;
        
        if ~isempty(TotalCH) && (length(TotalCH) > 1)
            numCH = length(TotalCH);
            clusterheadPositions = zeros(numCH, 2);
            for i = 1:numCH
                id = TotalCH(i).id;
                clusterheadPositions(i,:) = [Sensors(id).xd, Sensors(id).yd];
            end
            dists = distances(clusterheadPositions);
            thresholdDistance = 2;  
            if any(dists < thresholdDistance)
                proximity_penalty_dyn = proximity_penalty_dyn + penalty_increment;
                fprintf('Penalty applied: Clusterheads too close (min distance = %.2f); increasing proximity_penalty_dyn\n', min(dists));
            else
                proximity_penalty_dyn = max(proximity_penalty_dyn - penalty_increment, 0);
            end
        else
            proximity_penalty_dyn = max(proximity_penalty_dyn - penalty_increment, 0);
        end

        fprintf('Dynamic penalties: repeated=%.2f, proximity=%.2f, ch_ratio=%.2f\n', ...
            repeated_penalty_dyn, proximity_penalty_dyn, ch_ratio_penalty_dyn);
        
        loss = gcn_module.rl_train_step(gcnModel, X_py, A_py, py.float(RoundEnergy(r+1)), optimizer, ...
            py.float(repeated_penalty_dyn), py.float(proximity_penalty_dyn), py.float(ch_ratio_penalty_dyn));
        
        fprintf('Used Energy: %.4f\n', double(RoundEnergy(r+1)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        En=0;
        for i=1:n
            if Sensors(i).E>0
                En=En+(Sensors(i).E-AvgEnergyAllSensor(r+1))^2;
            end
        end
        
        Enheraf(r+1)=En/alive; %#ok
        
        title(sprintf('Round=%d,Dead nodes=%d', r+1, deadNum)) 
        
       %dead
       if(n==deadNum)
           
           lastPeriod=r;  
           break;
           
       end
      
    end % for r=0:1:rmax
    
    filename=sprintf('leach%d.mat',n);
    
    %% Save Report
    save(filename);
    
    gcnModel.save('trained_gcn_model.keras');
    fprintf('Modell gespeichert: trained_gcn_model.keras\n');
 end    