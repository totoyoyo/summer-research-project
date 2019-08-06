filenametemplate "%j-0%c.mps";

prologues:=3;

input TEX;
TEXPRE("%&latex" & char(10) & "\documentclass{article}\usepackage{amssymb,amsmath}\begin{document}");
TEXPOST("\end{document}");

u   := 1cm;          % scaling unit
lx  := 4u;           % number of units in positive x-axis
T   := 2u;
pair c;
c   := (lx,lx);      % center of picture
dashlen := u/6;

color clrB;          % Color for lenses Lambda
color clrb;          % Color for lenses lambda
clrB:= (.9,1,.9);
clrb:= (.9,.9,1);

picture domain;
picture BC[];
picture BC;
picture MC[];
picture NC[];

% domain
color axescolor;
axescolor := 0.9white;
axespen := 2bp;
xleft:=lx/5;
xright:=4lx/5;
t:= 4T/5;
%fill (xleft,0)--(xright,0)--(xright,t)--(xleft,t)--cycle withcolor axescolor;
draw (lx,0)--(lx,T)--(0,T) withcolor axescolor withpen pencircle scaled axespen;
drawarrow (-dashlen,0)--(lx+2dashlen,0) withcolor axescolor withpen pencircle scaled axespen;
drawarrow (0,-dashlen)--(0,T+2dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (lx,0)--(lx,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (xleft,0)--(xleft,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (xright,0)--(xright,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (0,T)--(-dashlen,T) withcolor axescolor withpen pencircle scaled axespen;
%draw (0,t)--(-dashlen,t) withcolor axescolor withpen pencircle scaled axespen;
label.rt (TEX("$x$"), (lx+2dashlen,0)) withcolor 0.5white;
label.bot (TEX("$0$"), (0,-dashlen)) withcolor 0.5white;
label.bot (TEX("$1$"), (lx,-dashlen)) withcolor 0.5white;
label.bot (TEX("$\eta_1$"), (xleft,-dashlen)) withcolor 0.5white;
label.bot (TEX("$\eta_2$"), (xright,-dashlen)) withcolor 0.5white;
label.top (TEX("$t$"), (0,T+2dashlen)) withcolor 0.5white;
label.lft (TEX("$0$"), (-dashlen,0)) withcolor 0.5white;
label.lft (TEX("$T$"), (-dashlen,T)) withcolor 0.5white;
%label.lft (TEX("$t$"), (-dashlen,t)) withcolor 0.5white;
domain := currentpicture;
currentpicture := nullpicture;

% BC
draw (0,0)--(0,T) withcolor 0.4green;
BC[1] := currentpicture;
draw (lx,0)--(lx,T) withcolor 0.4green;
BC[2] := currentpicture;
currentpicture := nullpicture;

% MC
beginfig(1);
	
	draw domain;
	draw (0,0)--(0,T) withcolor 0.4green;
	draw (xright,0)--(xright,T) withcolor 0.4green;
	draw (lx,0)--(lx,T) withcolor red;
	draw (xleft,0)--(xleft,T) withcolor red;
	
endfig;

% GR domains
beginfig(2);
	
	fill (xleft,0)--(xright,0)--(xright,t)--(xleft,t)--cycle withcolor clrB;
	fill (xright,0)--(lx,0)--(lx,t)--(xright,t)--cycle withcolor (1,.9,.9);
	fill (0,0)--(xleft,0)--(xleft,t)--(0,t)--cycle withcolor clrb;
	label (TEX("r=0"),(xleft/2,t/2));
	label (TEX("r=1"),((xleft+xright)/2,t/2));
	label (TEX("r=2"),((lx+xright)/2,t/2));
	draw domain;
	
endfig;

end





















