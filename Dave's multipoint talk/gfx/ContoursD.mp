filenametemplate "%j-0%c.mps";

prologues:=3;

input TEX;
TEXPRE("%&latex" & char(10) & "\documentclass{article}\usepackage{amssymb,amsmath}\begin{document}");
TEXPOST("\end{document}");

u   := 1cm;          % scaling unit
lx  := 3u;           % number of units in positive x-axis
pair c;
c   := (lx,lx);      % center of picture
dashlen := u/6;

picture wholepic;

% axes and contours
beginfig(1);

fill (0,0)--(2lx/3,2lx/3){dir 120}..(0,2lx/3)..(-2lx/3,2lx/3){dir 210}--cycle withcolor 0.9white;
fill ((0,0)--(2lx/3,2lx/3){dir 120}..(0,2lx/3)..(-2lx/3,2lx/3){dir 210}--cycle) rotated 180 withcolor 0.9white;

drawarrow (-lx,0)--(lx,0);
drawarrow (0,-2lx/3)--(0,2lx/3);
label.rt (TEX("$\operatorname*{Re}(\lambda)$"),(lx,0));
label.top (TEX("$\operatorname*{Im}(\lambda)$"),(0,2lx/3));

path Dplus;
Dplus := (-2lx/3,2lx/3)--(0,0)--(2lx/3,2lx/3);

draw subpath (1.5,2) of Dplus withcolor red;
drawarrow subpath (0.5,1.5) of Dplus withcolor red;
drawarrow subpath (0,0.5) of Dplus withcolor red;
draw (subpath (1.5,2) of Dplus) rotated 180 withcolor red;
drawarrow (subpath (0.5,1.5) of Dplus) rotated 180 withcolor red;
drawarrow (subpath (0,0.5) of Dplus) rotated 180 withcolor red;

label.urt (TEX("$D^+$"),(0,lx/3));
label.llft (TEX("$D^-$"),(0,-lx/3));
label.llft (TEX("$\partial D^+$"), point 0.5 of Dplus) withcolor red;
label.urt (TEX("$\partial D^-$"), point 0.5 of (Dplus rotated 180) ) withcolor red;


wholepic := currentpicture;
currentpicture := nullpicture;

draw wholepic shifted c;

endfig


end





















