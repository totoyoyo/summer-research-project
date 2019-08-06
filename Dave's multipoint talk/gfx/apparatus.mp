filenametemplate "%j-0%c.mps";

prologues:=3;

input TEX;
TEXPRE("%&latex" & char(10) & "\documentclass{article}\usepackage{amssymb,amsmath}\begin{document}");
TEXPOST("\end{document}");

beginfig(1);

u   := 1cm;          % scaling unit
lx  := 6u;           % number of units in positive x-axis
pair c;
c   := (lx,lx);      % center of picture

picture xaxis;
picture tube;
picture lamp;
picture cell;

dashlen := u/6;
tubewidth := u;
dashedlen := u;
lmpw := u;
lmpw2 := u/3;
lmph := u/4;
cellh := u/6;

% xaxis
drawarrow (-u/3,0)--(lx+u/3,0);
draw (0,0)--(0,-dashlen);
draw (lx,0)--(lx,-dashlen);
draw (lmpw,0)--(lmpw,-dashlen);
label.rt (TEX("$x$"),(lx+u/3,0));
label.bot (TEX("$0$"),(0,-dashlen));
label.bot (TEX("$1$"),(lx,-dashlen));
label.bot (TEX("$\vphantom{0}a\vphantom{0}$"),(lmpw,-dashlen));
xaxis := currentpicture;
currentpicture := nullpicture;

% tube
fill (lx,0)--(lx,tubewidth)--(-dashedlen,tubewidth){dir 220}..(-dashedlen,0){dir 220}--cycle withcolor 0.9white;
draw (-dashedlen,0)--(lx,0)--(lx,tubewidth)--(-dashedlen,tubewidth);
label (TEX("colloidal suspension"),(lx/2,tubewidth/2));
label.top (TEX("clear tube"),(3lx/4,tubewidth));
tube := currentpicture;
currentpicture := nullpicture;

% lamp
draw (subpath (4.8,7.2) of fullcircle scaled (u/4)) shifted (lmpw/2,u*cosd(360*0.15)/8);
draw (0,0)--(lmpw,0)--((lmpw+lmpw2)/2,lmph)--((lmpw-lmpw2)/2,lmph)--cycle;
label.rt (TEX("lamp"),(lmpw,lmph/2));
lamp := currentpicture;
currentpicture := nullpicture;

% cell
draw (0,0)--(lmpw,0)--(lmpw,cellh)--(0,cellh)--cycle;
label.rt (TEX("photovoltaic cell"),(lmpw,cellh/2));
cell := currentpicture;
currentpicture := nullpicture;

draw xaxis shifted c;
draw tube shifted (c+(0,2u/3));
draw lamp shifted (c+(0,u+tubewidth));
draw cell shifted (c+(0,u/4));

endfig;

beginfig(2);

picture domain;
picture BC;
picture MC;
picture NC;

% domain
draw (lx,0)--(lx,3u)--(0,3u) withcolor 0.8white withpen pencircle scaled 2.5bp;
drawarrow (-dashlen,0)--(lx+2dashlen,0) withcolor 0.8white withpen pencircle scaled 2.5bp;
drawarrow (0,-dashlen)--(0,3u+2dashlen) withcolor 0.8white withpen pencircle scaled 2.5bp;
draw (lx,0)--(lx,-dashlen) withcolor 0.8white withpen pencircle scaled 2.5bp;
draw (0,3u)--(-dashlen,3u) withcolor 0.8white withpen pencircle scaled 2.5bp;
label.rt (TEX("$x$"), (lx+2dashlen,0)) withcolor 0.5white;
label.bot (TEX("$0$"), (0,-dashlen)) withcolor 0.5white;
label.bot (TEX("$1$"), (lx,-dashlen)) withcolor 0.5white;
label.top (TEX("$t$"), (0,3u+2dashlen)) withcolor 0.5white;
label.lft (TEX("$0$"), (-dashlen,0)) withcolor 0.5white;
label.lft (TEX("$T$"), (-dashlen,3u)) withcolor 0.5white;
domain := currentpicture;
currentpicture := nullpicture;

% BC

BC := currentpicture;
currentpicture := nullpicture;

% MC

MC := currentpicture;
currentpicture := nullpicture;

% NC

NC := currentpicture;
currentpicture := nullpicture;

draw domain shifted c;

endfig;

end