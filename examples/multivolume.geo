Point(1) = {-20,-20,0.0};
Point(2) = {20,-20,0.0};
Point(3) = {20,0.0,0.0};
Point(4) = {-20,0.0,0.0};
Point(5) = {20,20,0.0};
Point(6) = {-20,20,0.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line(5) = {3,5};
Line(6) = {5,6};
Line(7) = {6,4};

Line Loop(1) = {1,2,3,4};
Line Loop(2) = {-3,5,6,7};

// Not sure why these need to be flipped
Plane Surface(1) = {-1};
Plane Surface(2) = {-2};

Physical Surface("Lower") = {1};
Physical Surface("Upper") = {2};

Physical Curve("Lower Sides") = {1,2,4};
Physical Curve("Upper Sides") = {5,6,7};
Physical Curve("Interface") = {3};

Mesh.MshFileVersion = 2.2;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

Mesh.ScalingFactor = 0.001;

mesh_scale = 16.;

min_size = 2/mesh_scale;
max_size = 2;

Mesh.CharacteristicLengthMin = min_size;
Mesh.CharacteristicLengthMax = max_size;

Field[1] = Distance;
Field[1].CurvesList = {3};
Field[1].NumPointsPerCurve = 100000;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = min_size;
Field[2].SizeMax = max_size;
Field[2].DistMin = 0.25;
Field[2].DistMax = 5;
Field[2].StopAtDistMax = 1;

Background Field = 2;
