filenametemplate "%j-0%c.mps";

prologues:=3;

input TEX;
TEXPRE("%&latex" & char(10) & "\documentclass{article}\usepackage{amssymb,amsmath}\begin{document}");
TEXPOST("\end{document}");

u   := 1cm;          % scaling unit
lx  := 5u;           % number of units in positive x-axis
T   := 2.5u;
pair c;
c   := (lx,lx);      % center of picture
dashlen := u/6;


picture domain;
picture BC[];
picture BC;
picture MC[];


% domain
color yncBlue;
yncBlue = 7red/255 + 77green/255 + 139blue/255; 
color axescolor;
axescolor := 0.9white;
axespen := 2bp;
%draw (lx,0)--(lx,T)--(0,T) withcolor axescolor withpen pencircle scaled axespen;
drawarrow (-dashlen,0)--(lx+2dashlen,0) withcolor axescolor withpen pencircle scaled axespen;
%drawarrow (0,-dashlen)--(0,T+2dashlen) withcolor axescolor withpen pencircle scaled axespen;
%draw (lx,0)--(lx,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
%draw (0,T)--(-dashlen,T) withcolor axescolor withpen pencircle scaled axespen;
label.rt (TEX("$x$"), (lx+2dashlen,0)) withcolor 0.5white;
label.bot (TEX("$0$"), (0,-dashlen)) withcolor 0.5white;
label.bot (TEX("$1$"), (lx,-dashlen)) withcolor 0.5white;
%label.top (TEX("$t$"), (0,T+2dashlen)) withcolor 0.5white;
%label.lft (TEX("$0$"), (-dashlen,0)) withcolor 0.5white;
%label.lft (TEX("$T$"), (-dashlen,T)) withcolor 0.5white;
domain := currentpicture;
currentpicture := nullpicture;

%\definecolor{yncBlue}{RGB}{7, 77, 139}
% BC
draw (0,0)--(0,T) withcolor 0.4green;
BC[1] := currentpicture;
draw (lx,0)--(lx,T) withcolor 0.4green;
BC[2] := currentpicture;
currentpicture := nullpicture;

% MC
MCw := lx;
for j=2 step 1 until 8:
	for i=0 step 1/(j-1) until 1+0.5/(j-1):
		draw (i*MCw,0) withcolor yncBlue withpen pencircle scaled 4bp; %--(i*MCw,1)
	endfor;
	MC[j-1] := currentpicture;
	currentpicture := nullpicture;
endfor;


% domain only: figure 1
%beginfig(1); draw domain shifted c; endfig;
% BC: figures 2-3
%for i=2 upto 3: beginfig(i); draw domain shifted c; draw BC[i-1] shifted c; endfig; endfor;
% MC: figures 4-10
for i=4 upto 10: beginfig(i); draw domain shifted c; draw MC[i-3] shifted c; endfig; endfor;
% MC: figures 11-16
%for i=11 upto 16: beginfig(i); draw domain shifted c; draw NC[i-10] shifted c; endfig; endfor;

end





















