x0=1.0/6.0;
setsize=0.025;

Point(1) = {0, 0, 0, setsize};
Point(2) = {x0,0, 0, setsize};
Point(3) = {4, 0, 0, setsize};
Point(4) = {4, 1, 0, setsize};
Point(5) = {0, 1, 0, setsize};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(5) = {3, 4};
Line(6) = {4, 5};
Line(7) = {5, 1};

Line Loop(8) = {-5, -6, -7, -1, -2};
Plane Surface(8) = {8};

Physical Surface('domain') = {8};
Physical Curve('ic1') = {6};
Physical Curve('ic2') = {7};
Physical Curve('ic3') = {1};
Physical Curve('wall') = {2};
Physical Curve('out') = {5};
