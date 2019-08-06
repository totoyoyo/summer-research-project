filenametemplate "%j-0%c.mps";

prologues:=3;

input TEX;
TEXPRE("%&latex" & char(10) & "\documentclass{article}\usepackage{amssymb,amsmath}\begin{document}");
TEXPOST("\end{document}");

u   := 1cm;          % scaling unit
lx  := 8u;           % number of units in positive x-axis
T   := 4u;
pair c;
c   := (lx,lx);      % center of picture
dashlen := u/6;



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
fill (xleft,0)--(xright,0)--(xright,t)--(xleft,t)--cycle withcolor axescolor;
draw (lx,0)--(lx,T)--(0,T) withcolor axescolor withpen pencircle scaled axespen;
drawarrow (-dashlen,0)--(lx+2dashlen,0) withcolor axescolor withpen pencircle scaled axespen;
drawarrow (0,-dashlen)--(0,T+2dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (lx,0)--(lx,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (xleft,0)--(xleft,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (xright,0)--(xright,-dashlen) withcolor axescolor withpen pencircle scaled axespen;
draw (0,T)--(-dashlen,T) withcolor axescolor withpen pencircle scaled axespen;
draw (0,t)--(-dashlen,t) withcolor axescolor withpen pencircle scaled axespen;
label.rt (TEX("$x$"), (lx+2dashlen,0)) withcolor 0.5white;
label.bot (TEX("$0$"), (0,-dashlen)) withcolor 0.5white;
label.bot (TEX("$1$"), (lx,-dashlen)) withcolor 0.5white;
label.bot (TEX("$y$"), (xleft,-dashlen)) withcolor 0.5white;
label.bot (TEX("$z$"), (xright,-dashlen)) withcolor 0.5white;
label.top (TEX("$t$"), (0,T+2dashlen)) withcolor 0.5white;
label.lft (TEX("$0$"), (-dashlen,0)) withcolor 0.5white;
label.lft (TEX("$T$"), (-dashlen,T)) withcolor 0.5white;
label.lft (TEX("$t$"), (-dashlen,t)) withcolor 0.5white;
domain := currentpicture;
currentpicture := nullpicture;

% BC
draw (0,0)--(0,T) withcolor 0.4green;
BC[1] := currentpicture;
draw (lx,0)--(lx,T) withcolor 0.4green;
BC[2] := currentpicture;
currentpicture := nullpicture;

% domain and Green's theorem to derive GR
beginfig(1);
	
	draw domain shifted c;
	label (TEX("$\displaystyle +\int_y^zX(x,0,\lambda)\mathrm{d} x$"),((xleft+xright)/2,0)) shifted c;
	label (TEX("$\displaystyle -\int_y^zX(x,t,\lambda)\mathrm{d} x$"),((xleft+xright)/2,t)) shifted c;
	label (TEX("$\displaystyle +\int_0^tY(z,s,\lambda)\mathrm{d} s$"),(xright,t/2)) shifted c;
	label (TEX("$\displaystyle -\int_0^tY(y,s,\lambda)\mathrm{d} s$"),(xleft,t/2)) shifted c;
	
	
endfig


end





















