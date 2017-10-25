% The code contained in this file originated in the CMU Suter Research
% Group with contributions from Frankie Li, Jonathan Lind, David Menasche, 
% Rulin Chen, RM Suter, and others.

% A set of utility functions (rotation representation conversions) from the 
% Cornell University group of Paul Dawson et al is used and appears at the 
% end of the file.

% This is a stand-alone set of functions that plot information in
% .mic files

% Usage information can be seen by typing 'plot_mic([0])'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%  plot_mic.m
  %%
  %%  Usage:
  %%  plot_mic( snp, sidewidth, plotType, ConfidenceRange, Symmetry, NewFigure, bCenter, bTightFit, ConfColorScaleRange, ConfColorScaleType )
  %%
  %%
  %%  snp - a snapshot read from a .mic file using load_mic
  %%  sidewidth - the width of one side of a hexagon returned by load_mic
  %%  plotType - 1 => plots of orientations, 2 => plots of confidence level
  %%
  %%  Note that there is limitations on the colormap resolution.
  %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%
  %% .mic File Format:
  %% Col 1-3 x, y, z
  %% Col 4   1 = triangle pointing up, 2 = triangle pointing down
  %% Col 5 generation number; triangle size = sidewidth /(2^generation number )
  %% Col 6 Phase - 1 = exist, 0 = not fitted
  %% Col 7-9 orientation
  %% Col 10  Confidence
  %%
  %% There may be other information in additions columns
  %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  function plot_mic( varargin )

% Suter group

%  figure(3);
%    Symmetry = 'NON'
%    Symmetry = 'CUB'
%    Symmetry = 'HEX';

    degrad = 180./pi;       

%%%%% Interpret input variables  
  if ( nargin < 5 )
    DisplayUsage();
    error('ERROR: Not enough arguments: first five are required.');
  end

  ValidPlotType = [1, 2, 3, 4, 5, 6 ,9];

  snp       = varargin{1};
  sidewidth = varargin{2};
  plotType  = varargin{3};
  ConfidenceRange = varargin{4};
  %if(varargin{5} == 1)
  %    Symmetry = 'CUB'
  %elseif(varargin{5} == 2)
  %    Symmetry = 'HEX'
  %else
  %    Symmetry = 'NON'
  %end
  Symmetry = varargin{5};

  %% default parameters
  NewFigure = false;
  bCenter   = false;
  bTightFit = true;

  
  if( nargin >= 6 )
      NewFigure = varargin{6};
      if(NewFigure)
          figure();
      end
      % dz = varargin{5};
  end

  if( nargin >= 7 )
    bCenter = varargin{7};
  end
    
  if( nargin >= 8 )
    bTightFit = varargin{8};
  end
  
  if( nargin >= 9 )
    ConfColorScale = varargin{9};
  end
  
  if( nargin >= 10 )
    ColorType = varargin{10};
  else
    ColorType = 'Red';
  end
  
  %if( plotType < 1 & plotType > 9 )
  %  DisplayUsage();
  %  error('Plot Type not defined!  See USAGE for detail');
  %end

%%%%% Select entries with confidence in range
  ConfSize = size( ConfidenceRange );
  if( ConfSize(1) > 1 || ConfSize(2) > 1 )
	findvec = snp(:, 10) >= ConfidenceRange(1)  & snp(:, 10) <= ConfidenceRange(2);
  else
    findvec = snp(:, 10) >= ConfidenceRange ;
  end
  snp = snp(findvec, :);

%%%%% Extract vectors
  if(plotType~=9)
      % GetTriangleVertices is below in this file
      [tri_x, tri_y, snp_Euler, snp_Conf ] = GetTriangleVertices( snp, sidewidth );
  else 
      % Requires external GetTriangleVerticesGrainID (Lind??)
      [tri_x, tri_y, snp_Euler, snp_Conf ] = GetTriangleVerticesGrainID( snp, sidewidth );
  end
 
  
%%%%%%%%%% The next section selects plot types and puts plot quantities in 'tmp' 
  
%%%%% Plot grain map (scaled Eulers?)
  if(plotType == 1)
    tmp =  [ snp_Euler ]/480;
    tmp = [1-tmp(:, 1), tmp(:, 2), 1-tmp(:, 3)];
    
%%%%% Plot Confidence map; options for range and color scale type
  elseif(plotType == 2)
    if( nargin < 8 )
      tmp = (snp_Conf - min(snp_Conf));
      tmp = tmp/max(tmp);
      tmp = [tmp, zeros(length(tmp), 1), 1-tmp];
      
      xConf = [0:0.01:1]';
      colormap( [ xConf, xConf * 0, 1- xConf]  );
      caxis( [min(snp_Conf), max(snp_Conf) ] );
      %caxis( [min(snp_Conf), max(snp_Conf) ] );
    else  % clipping
      tmp = [ snp_Conf ];
      tmp = max(tmp, ConfColorScale(1));
      tmp = min(tmp, ConfColorScale(2) );

%       tmp = [ snp_Conf ] - ConfColorScale(1);   % original auto rescale
      % original auto rescale
      disp('rescale');
      tmp = tmp - min(tmp);

      tmp = tmp /( max(tmp) - min(tmp) ); 
      if( strcmp( ColorType, 'Red') )
          tmp = [tmp, zeros(length(tmp), 1), 1-tmp];
          xConf = [ min(tmp):0.01:max(tmp)]';
          colormap( [ xConf, xConf * 0, 1- xConf]  );
      elseif( strcmp(ColorType, 'Gray') )
          tmp = [tmp , tmp, tmp];
          xConf = [ min(tmp):0.01:max(tmp)]';
          colormap( [ xConf, xConf , xConf]  );
      elseif( strcmp(ColorType, 'iGray') )
          tmp = [max(tmp) - tmp , max(tmp) - tmp, max(tmp) - tmp];
          xConf = [ min(tmp):0.01:max(tmp)]';
          cmap = [ xConf, xConf , xConf];
          colormap( flipud(cmap)  );
      else
          error('Unknown colorscale type');
      end
      caxis( [ConfColorScale(1),ConfColorScale(2)] );
    end
     % colorbar( 'location', 'eastoutside');


%%%%% plot Rodrigues vector orientation map (with scaling options)
  elseif(plotType==3) 
    tmp = RodOfQuat( QuatOfRMat( 	RMatOfBunge(snp_Euler', 'degrees') ) )';
    InRange = tmp(:, 1) > -10e5 & tmp(:, 1) < 10e5 ...
      & tmp(:, 2) > -10e5 & tmp(:, 2) < 10e5 ...
      & tmp(:, 3) > -10e5 & tmp(:, 3) < 10e5;
    
    tmp2 = tmp( InRange, :);

%%% Set R-vector limits by symmetry about x,y,z axes

    if(Symmetry == 'CUB')
        %Scale by size of CUBIC FZ (fixes color scale for whole range)
        disp('Color scaling to Cubic symmetry FZ')
    
        sym = 90.
        f = tan(sym/degrad/4);
        disp(['R_1, R_2, R_3: Sym angle = ', num2str(sym),', Rvec range = +/-', num2str(f)])
        tmp2(:, 1) = (tmp2(:, 1) + f)/(2*f); 
        big = max(tmp2(:,1));
        small = min(tmp2(:,1));
        disp(['Color range in data: Max = ', num2str(big),', Min = ',num2str(small)])
        tmp2(:, 2) = (tmp2(:, 2) + f)/(2*f);
        big = max(tmp2(:,1));
        small = min(tmp2(:,1));
        disp(['Color range in data: Max = ', num2str(big),', Min = ',num2str(small)])
        tmp2(:, 3) = (tmp2(:, 3) + f)/(2*f);
        big = max(tmp2(:,1));
        small = min(tmp2(:,1));
        disp(['Color range in data: Max = ', num2str(big),', Min = ',num2str(small)])

    % Scale by size of HEXAGONAL FZ (fixes color scale for whole range)

    elseif(Symmetry == 'HEX')

        disp('Color scaling to Hexagonal symmetry FZ')
    
        sym = 180.;                 % Two fold
        f = tan(sym/degrad/4);
        disp(['R_1, R_2: Sym angle = ', num2str(sym),', Rvec range = +/-', num2str(f)])
        tmp2(:,1) = (tmp2(:,1) + f)/(2*f);
        big = max(tmp2(:,1));
        small = min(tmp2(:,1));
        disp(['Color range in data: Max = ', num2str(big),', Min = ',num2str(small)])

        tmp2(:,2) = (tmp2(:,2) + f)/(2*f);
        big = max(tmp2(:,2));
        small = min(tmp2(:,2));
        disp(['Color range in data: Max = ', num2str(big),', Min = ',num2str(small)])

        disp(' ')
        sym = 60.;                 % Six fold about c-axis
        f = tan(sym/degrad/4);
        disp(['R_3: Sym angle = ',num2str(sym),', Rvec range = +/-',num2str(f)])    
        tmp2(:,3) = (tmp2(:,3) + f)/(2*f);
        big = max(tmp2(:,3));
        small = min(tmp2(:,3));
        disp(['Color range in data: Max = ', num2str(big), ', Min = ', num2str(small)])
    
        tmp( InRange, :) = tmp2;  
    else
        disp('Color scaling to input range (no symmetry consideration)')
        
        %Scale by INPUT RANGE used (expand scale for seeing subtle stuff)
        tmp2(:, 1) =  tmp2(:, 1) - min(tmp2(:, 1));
        tmp2(:, 2) =  tmp2(:, 2) - min(tmp2(:, 2));
        tmp2(:, 3) =  tmp2(:, 3) - min(tmp2(:, 3));
        
        tmp2(:,1) = tmp2(:,1)/max(tmp2(:,1));
        tmp2(:,2) = tmp2(:,2)/max(tmp2(:,2));
        tmp2(:,3) = tmp2(:,3)/max(tmp2(:,3));
        
        tmp( InRange, :) = tmp2;
    end


%%%%%
  elseif(plotType == 4) % plot pole figure
    tmp = fixedEulerColor( snp_Euler );


%%%%%
  elseif(plotType == 5)
    disp('IPF');
    tmp = inversePoleFigureColor([ snp_Euler ] * pi/180);


%%%%% Plot the mesh
  elseif(plotType == 6)

    tmp =  [snp_Euler]/480;
    tmp = [1-tmp(:, 1), tmp(:, 2), 1-tmp(:, 3)];
    size_vec = size(tmp);
    tri_color = ones(1, size_vec(1), size_vec(2));
    % figure;
    axis equal;
    
    % do this plot here:
    h = patch(tri_x, tri_y, tri_color);
    
    
%%%%% Misorientation map
  elseif(plotType == 7) 

    tmp = [ snp_Conf ];
    tmp = tmp/max(tmp);
    tmp = [tmp, zeros(length(tmp), 1), 1-tmp];

    step = ( max(snp_Conf) - min(snp_Conf) ) /10;
    xConf = [min(snp_Conf):step:max(snp_Conf)]';
    xConf = xConf / max(xConf);
    colormap( [ xConf, xConf * 0, 1- xConf]  );
    caxis( [min(snp_Conf), max(snp_Conf) ] );
    colorbar( 'location', 'eastoutside');
    
    
%%%%%
  elseif(plotType == 8) % plot Euler angle color map (grain map) 
  
% set color scale in [0:1]
    tmp =   snp_Euler ;
   
    tmp(:, 1) =  tmp(:, 1) - min(tmp(:, 1))  ;
    tmp(:, 2) =  tmp(:, 2) - min(tmp(:, 2))  ;
    tmp(:, 3) =  tmp(:, 3) - min(tmp(:, 3))  ;


    tmp(:, 3) = tmp(:, 3) / max(tmp(:, 3));
    tmp(:, 2) = tmp(:, 2) / max(tmp(:, 2));
    tmp(:, 1) = tmp(:, 1) / max(tmp(:, 1));
    

%%%%%    
  elseif(plotType == 9) % plot GrainIDs
      if( nargin < 8 )
        tmp = [ snp_Conf ];
        tmp = [ snp_Conf ] - min( snp_Conf );   % original auto rescale
        tmp = tmp /( max(snp_Conf) - min(snp_Conf) );  % original auto rescale
        tmp = [tmp, zeros(length(tmp), 1), 1-tmp];
        xConf = [0:0.01:1]';
        colormap( [ xConf, xConf * 0, 1- xConf]  );
        caxis( [min(snp_Conf), max(snp_Conf) ] );   % original, auto rescale
        caxis( [min(snp_Conf), max(snp_Conf) ] );
      else  % clipping
        
        tmp = [ snp_Conf ];
        tmp = max(tmp, ConfColorScale(1));
        tmp = min(tmp, ConfColorScale(2) );

%         tmp = [ snp_Conf ] - ConfColorScale(1);   % original auto rescale
        	 % original auto rescale
        disp('rescale');
        tmp = tmp - min(tmp);

        tmp = tmp /( max(tmp) - min(tmp) ); 
        if( strcmp( ColorType, 'Red') )
          tmp = [tmp, zeros(length(tmp), 1), 1-tmp];
          xConf = [ min(tmp):0.01:max(tmp)]';
          colormap( [ xConf, xConf * 0, 1- xConf]  );
        elseif( strcmp(ColorType, 'Gray') )
          tmp = [tmp , tmp, tmp];
          xConf = [ min(tmp):0.01:max(tmp)]';
          colormap( [ xConf, xConf , xConf]  );
        elseif( strcmp(ColorType, 'iGray') )
          tmp = [max(tmp) - tmp , max(tmp) - tmp, max(tmp) - tmp];
          xConf = [ min(tmp):0.01:max(tmp)]';
          cmap = [ xConf, xConf , xConf];
          colormap( flipud(cmap)  );
        else
          error('Unknown type');
        end
        caxis( [ConfColorScale(1),ConfColorScale(2)] );   % original, auto rescal 
      end
      
      nMinSize = 100;
      gids = unique(snp(:,11));
      for i=1:length(gids)
          ind = find( snp(:,11) == gids(i) );
          if size(ind,1) > nMinSize
              x_avg = mean( snp(ind,1) );
              y_avg = mean( snp(ind,2) );
              text(x_avg,y_avg, num2str(gids(i)), 'Color', 'w','FontSize',10);
          end
      end  
      colorbar( 'location', 'eastoutside');
  end 
  
  
%%%%% Fill color arrays from tmp and shift origin and axes if requested (except for mesh plotting)
  if( plotType ~= 6 )
    size_vec = size(tmp);
    tri_color = zeros(1, size_vec(1), size_vec(2));
    tri_color(1, :, :) = tmp;
    %figure;
    
    if ~bCenter 
      if ( ~bTightFit )
       %axis([ -sidewidth, sidewidth, -sidewidth, sidewidth]);
        axis([ -sidewidth, sidewidth, -sidewidth, sidewidth]);
      else
        %micCenter = [mean(snp(:, 1)), mean( snp(:, 2) )];
        micCenter = [mean(tri_x(:)), mean( tri_y(:))];
        dx = max( tri_x(:) ) - min( tri_x(:) );
        dy = max( tri_y(:) ) - min( tri_y(:) );
        
        newSw = 1.1*max( [dx, dy] ) / 2;             % adjust 1.1 fudge factor for more asymmetric cases than Zr
       %axis([ micCenter(1) - newSw, micCenter(1) + newSw, micCenter(2) - newSw, micCenter(2) + newSw ] );
        axis([ micCenter(1) - newSw, micCenter(1) + newSw, micCenter(2) - newSw, micCenter(2) + newSw ] );
      end
    else
      findvec = find( snp(:, 6) > 0 );
      xCenter = mean( snp(findvec, 1) );
      yCenter = mean( snp(findvec, 2) );
     %axis([ -sidewidth + xCenter, sidewidth + xCenter, -sidewidth + yCenter, sidewidth + yCenter]);
      axis([ -sidewidth + xCenter, sidewidth + xCenter, -sidewidth + yCenter, sidewidth + yCenter]);
    end

  end

%%%%% Make the plot!
  %if( nargin >= 5 )
    %  zs = zeros(3,length(tri_x))+dz;
    %  %size(tri_x)
    %  %size(zs)
    %  %size(tri_color)
    %  h = patch(tri_x, tri_y, zs,  tri_color, 'EdgeColor', 'none');
    %  %alpha(h,1);
    % else
    h = patch(tri_x, tri_y, tri_color, 'EdgeColor', 'none');        
    %end
  if(plotType==2)
      colorbar( 'location', 'eastoutside');
  end    
  xlabel('X (mm)', 'FontSize', 20);
  ylabel('Y (mm)', 'FontSize', 20);
  set(gca , 'FontSize', 16);
  box on;
 %axis square;
  axis equal;
  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%
  %%  GetTriangleVertices
  %%
  %%   Compute all vertices given a legacy snapshot format
  %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 function [tri_x, tri_y, snp_Euler, snp_Conf ] = GetTriangleVertices( snp, sidewidth )

 % Suter group

    snp(:,5) = 2.^ snp(:, 5);

    % find range
    max_x = 1*sidewidth;
    max_y = 1*sidewidth;
    min_x = -1*sidewidth;
    min_y = -1*sidewidth;

    % find all fitted triangles
    snp = sortrows(snp, 6);
    findvec = find(snp(:, 6) > 0);
    snp = snp(findvec, :);

    % sort by triangle
    snp = sortrows(snp, 4);

    % find triangles pointing up
    downsIndex = find(snp(:, 4) > 1);
    upsIndex = find(snp(:, 4) <= 1);

    ups = snp(upsIndex, :);
    downs = snp(downsIndex, :);

    ups_sides = sidewidth ./ ups(:, 5);
    downs_sides = sidewidth ./ downs(:, 5);

    % calculate ups v1, ups v2, and ups v3
    ups_v1 = ups(:, 1:2);      % (x, y)
    ups_v2 = ups(:, 1:2);
    ups_v2(:, 1) = ups_v2(:, 1) + ups_sides;  % (x+s, y) direction
    ups_v3 = ups(:, 1:2);
    ups_v3(:, 1) = ups_v3(:, 1) + ups_sides/2; % (x+s/2, y)
    ups_v3(:, 2) = ups_v3(:, 2) + ups_sides/2 * sqrt(3); % (x+s/2, y+s/2 *sqrt(3));


    % calculate downs v1, downs v2, and downs v3
    downs_v1 = downs(:, 1:2);      % (x, y)
    downs_v2 = downs(:, 1:2);
    downs_v2(:, 1) = downs_v2(:, 1) + downs_sides;  % (x+s, y) direction
    downs_v3 = downs(:, 1:2);
    downs_v3(:, 1) = downs_v3(:, 1) + downs_sides/2; % (x+s/2, y)
    downs_v3(:, 2) = downs_v3(:, 2) - downs_sides/2 * sqrt(3); % (x+s/2, y - s/2 *sqrt(3));

    % format is in [v1; v2; v3], where v1, v2, and v3 are rol vectors
    tri_x = [ [ups_v1(:, 1); downs_v1(:, 1)]'; [ups_v2(:, 1); downs_v2(:, 1)]'; [ups_v3(:, 1); downs_v3(:, 1)]'];
    tri_y = [ [ups_v1(:, 2); downs_v1(:, 2)]'; [ups_v2(:, 2); downs_v2(:, 2)]'; [ups_v3(:, 2); downs_v3(:, 2)]'];

    snp_Reordered = [ ups; downs];
    snp_Euler = snp_Reordered( :, 7:9 );
%   snp_Conf  = snp_Reordered( :, 10  );
    snp_Conf  = snp_Reordered( :, 10  );
%   snp_Conf  = snp_Reordered( :, 13  ) ./snp_Reordered( :, 14  );
end



  function [tri_x, tri_y, snp_Euler, snp_GID ] = GetTriangleVerticesGrainID( snp, sidewidth )

  % Suter group

  snp(:,5) = 2.^ snp(:, 5);

  % find range
  max_x = 1*sidewidth;
  max_y = 1*sidewidth;
  min_x = -1*sidewidth;
  min_y = -1*sidewidth;

  % find all fitted triangles
  snp = sortrows(snp, 6);
  findvec = find(snp(:, 6) > 0);
  snp = snp(findvec, :);

  % sort by triangle
  snp = sortrows(snp, 4);

  % find triangles pointing up
  downsIndex = find(snp(:, 4) > 1);
  upsIndex = find(snp(:, 4) <= 1);

  ups = snp(upsIndex, :);
  downs = snp(downsIndex, :);


  ups_sides = sidewidth ./ ups(:, 5);
  downs_sides = sidewidth ./ downs(:, 5);

  % calculate ups v1, ups v2, and ups v3
  ups_v1 = ups(:, 1:2);      % (x, y)
  ups_v2 = ups(:, 1:2);
  ups_v2(:, 1) = ups_v2(:, 1) + ups_sides;  % (x+s, y) direction
  ups_v3 = ups(:, 1:2);
  ups_v3(:, 1) = ups_v3(:, 1) + ups_sides/2; % (x+s/2, y)
  ups_v3(:, 2) = ups_v3(:, 2) + ups_sides/2 * sqrt(3); % (x+s/2, y+s/2 *sqrt(3));


  % calculate downs v1, downs v2, and downs v3
  downs_v1 = downs(:, 1:2);      % (x, y)
  downs_v2 = downs(:, 1:2);
  downs_v2(:, 1) = downs_v2(:, 1) + downs_sides;  % (x+s, y) direction
  downs_v3 = downs(:, 1:2);
  downs_v3(:, 1) = downs_v3(:, 1) + downs_sides/2; % (x+s/2, y)
  downs_v3(:, 2) = downs_v3(:, 2) - downs_sides/2 * sqrt(3); % (x+s/2, y - s/2 *sqrt(3));

  % format is in [v1; v2; v3], where v1, v2, and v3 are rol vectors
  tri_x = [ [ups_v1(:, 1); downs_v1(:, 1)]'; [ups_v2(:, 1); downs_v2(:, 1)]'; [ups_v3(:, 1); downs_v3(:, 1)]'];
  tri_y = [ [ups_v1(:, 2); downs_v1(:, 2)]'; [ups_v2(:, 2); downs_v2(:, 2)]'; [ups_v3(:, 2); downs_v3(:, 2)]'];

  snp_Reordered = [ ups; downs];
  snp_Euler = snp_Reordered( :, 7:9 );
%   snp_Conf  = snp_Reordered( :, 10  );
   snp_GID  = snp_Reordered( :, 11  );
%  snp_Conf  = snp_Reordered( :, 13  ) ./snp_Reordered( :, 14  );
 end 



  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%
  %%  DisplayUsage
  %%
  %%
  %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function DisplayUsage()

% Suter group
  
  disp(' ')
    disp('Use [SnapShotName, SideWidth] = LOAD_MIC(mic_filename, #columns) to load data')
    disp(' ')
    disp('USAGE:  plot_mic( SnapShotName, SideWidth, PlotType, Confidence{min or [min max]}, Symmetry{"CUB", "HEX","NON"},'),
    disp('        NewFigure{default=false}, bCenter{default=false}, bTightFit{default=true}, ConfColorScaleRange, ConfColorScaleType )')
    disp('             (first five arguments are required)') 
    disp(' ')
    disp('        PlotType = 1 -- Direct mapping of Euler angles to RGB color (not recommended)')
    disp('        PlotType = 2 -- Confidence map')
    disp('        PlotType = 3 -- Map of RF-vectors represented by RGB colors')
    disp('        PlotType = 5 -- Inverse Pole Figure (implementation not verified)')
    disp('        PlotType = 6 -- Mesh Structure')
  end


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%
  %%  fixedEulerColor
  %%
  %%
  %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Suter group
  
  function colorOut = fixedEulerColor(Eu)

    colorOut = Eu;
    EuSize = size(Eu);
    EuLength = EuSize(1);

    % calculate misorientation from fixed color and vectors (cubic symmetry)

    %
    %  (35, 45, 0)
    %  (90, 35, 45)
    %  (42, 37, 9)
    %  (59, 37, 63)
    %  (0, 45, 0)
    %  (0, 0, 0)
    %  (90, 25, 65)

    colorList = cell(7, 2);
    colorList{1, 1} = [35, 45, 0];
    colorList{2, 1} = [90, 35, 45];
    colorList{3, 1} = [42, 37, 9];
    colorList{4, 1} = [59, 37, 63];
    colorList{5, 1} = [0, 45, 0];
    colorList{6, 1} = [0, 0, 0];
    colorList{7, 1} = [90, 25, 65];


    colorList{1, 2} = [.5, 0, 0];
    colorList{2, 2} = [0, .5, 0];
    colorList{3, 2} = [0, 0, .5];
    colorList{4, 2} = [.5, .5, 0];
    colorList{5, 2} = [0, .5, 1];
    colorList{6, 2} = [.5, 0, .5];
    colorList{7, 2} = [.5, .5, .5];


    allowedMisorient = 15;

    cubicSymOps = GetCubicSymOps();

    test = zeros(EuLength, 2);
    for i = 1:EuLength

        minMisorient = 360;
        minJ = -1;
        for j = 1:7
            currentMisorient = misorient_sym_deg(Eu(i, :), colorList{j, 1}, cubicSymOps, 24);
            if(currentMisorient < minMisorient)
                minMisorient = currentMisorient;
                minJ = j;
            end
        end

        test(i, 1) = minJ;
        test(i, 2) = minMisorient;
        if(minJ > -1 & minMisorient < allowedMisorient)
            colorOut(i, :) = colorList{minJ, 2};
        else
            colorOut(i, :) = [0, 0, 0];
        end
    end
    disp('end');
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function rmat = RMatOfBunge(bunge, units)
  
  % From Cornell OdfPf package

% RMatOfBunge - Rotation matrix from Bunge angles.
%
%   USAGE:
%
%   rmat = RMatOfBunge(bunge, units)
%
%   INPUT:
%
%   bunge is 3 x n,
%         the array of Bunge parameters
%   units is a string,
%         either 'degrees' or 'radians'
%       
%   OUTPUT:
%
%   rmat is 3 x 3 x n,
%        the corresponding rotation matrices
%   
if (nargin < 2)
  error('need second argument, units:  ''degrees'' or ''radians''')
end
%
if (strcmp(units, 'degrees'))
  %
  indeg = 1;
  bunge = bunge*(pi/180);
  %
elseif (strcmp(units, 'radians'))
  indeg = 0;
else
  error('angle units need to be specified:  ''degrees'' or ''radians''')
end
%
n    = size(bunge, 2);
cbun = cos(bunge);
sbun = sin(bunge);
%
rmat = [
     cbun(1, :).*cbun(3, :) - sbun(1, :).*cbun(2, :).*sbun(3, :);
     sbun(1, :).*cbun(3, :) + cbun(1, :).*cbun(2, :).*sbun(3, :);
     sbun(2, :).*sbun(3, :);
    -cbun(1, :).*sbun(3, :) - sbun(1, :).*cbun(2, :).*cbun(3, :);
    -sbun(1, :).*sbun(3, :) + cbun(1, :).*cbun(2, :).*cbun(3, :);
     sbun(2, :).*cbun(3, :);
     sbun(1, :).*sbun(2, :);
    -cbun(1, :).*sbun(2, :);
     cbun(2, :)
    ];
rmat = reshape(rmat, [3 3 n]);
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function quat = QuatOfRMat(rmat)

% From Cornell OdfPf package

% QuatOfRMat - Quaternion from rotation matrix
%   
%   USAGE:
%
%   quat = QuatOfRMat(rmat)
%
%   INPUT:
%
%   rmat is 3 x 3 x n,
%        an array of rotation matrices
%
%   OUTPUT:
%
%   quat is 4 x n,
%        the quaternion representation of `rmat'

% 
%  Find angle of rotation.
%
ca = 0.5*(rmat(1, 1, :) + rmat(2, 2, :) + rmat(3, 3, :) - 1);
ca = min(ca, +1);
ca = max(ca, -1);
angle = squeeze(acos(ca))';
%
%  Three cases for the angle:  
%  
%  *   near zero -- matrix is effectively the identity
%  *   near pi   -- binary rotation; need to find axis
%  *   neither   -- general case; can use skew part
%
tol = 1.0e-4;
anear0 = (angle < tol);
nnear0 = length(anear0);
angle(anear0) = 0;

raxis = [rmat(3, 2, :) - rmat(2, 3, :);
	rmat(1, 3, :) - rmat(3, 1, :);
	rmat(2, 1, :) - rmat(1, 2, :)];
raxis = squeeze(raxis);
raxis(:, anear0) = 1;
%
special = angle > pi - tol;
nspec   = sum(special);
if (nspec > 0)
  %disp(['special: ', num2str(nspec)]);
  sangle = repmat(pi, [1, nspec]);
  tmp = rmat(:, :, special) + repmat(eye(3), [1, 1, nspec]);
  tmpr = reshape(tmp, [3, 3*nspec]);
  tmpnrm = reshape(dot(tmpr, tmpr), [3, nspec]);
  [junk, ind] = max(tmpnrm);
  ind = ind + (0:3:(3*nspec-1));
  saxis = squeeze(tmpr(:, ind));
  raxis(:, special) = saxis;
end
%debug.angle = angle;
%debug.raxis = raxis;
quat = QuatOfAngleAxis(angle, raxis);  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rod = RodOfQuat(quat)

% From Cornell OdfPf package

% RodOfQuat - Rodrigues parameterization from quaternion.
%   
%   USAGE:
%
%   rod = RodOfQuat(quat)
%
%   INPUT:
%
%   quat is 4 x n, 
%        an array whose columns are quaternion paramters; 
%        it is assumed that there are no binary rotations 
%        (through 180 degrees) represented in the input list
%
%   OUTPUT:
%
%  rod is 3 x n, 
%      an array whose columns form the Rodrigues parameterization 
%      of the same rotations as quat
% 
rod = quat(2:4, :)./repmat(quat(1,:), [3 1]);
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function quat = QuatOfAngleAxis(angle, raxis)
% QuatOfAngleAxis - Quaternion of angle/axis pair.

% From Cornell OdfPf package
%
%   USAGE:
%
%   quat = QuatOfAngleAxis(angle, rotaxis)
%
%   INPUT:
%
%   angle is an n-vector, 
%         the list of rotation angles
%   raxis is 3 x n, 
%         the list of rotation axes, which need not
%         be normalized (e.g. [1 1 1]'), but must be nonzero
%
%   OUTPUT:
%
%   quat is 4 x n, 
%        the quaternion representations of the given
%        rotations.  The first component of quat is nonnegative.
%   
halfangle = 0.5*angle(:)';
cphiby2   = cos(halfangle);
sphiby2   = sin(halfangle);
%
rescale = sphiby2 ./sqrt(dot(raxis, raxis, 1));
%
quat = [cphiby2; repmat(rescale, [3 1]) .* raxis ] ;
%
q1negative = (quat(1,:) < 0);
quat(:, q1negative) = -1*quat(:, q1negative);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
