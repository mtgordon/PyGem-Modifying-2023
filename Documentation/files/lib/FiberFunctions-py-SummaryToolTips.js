﻿NDSummary.OnToolTipsLoaded("File:lib/FiberFunctions.py",{174:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype174\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> find_index_of_element_connectons(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">content</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div></div>",175:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype175\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> try_int(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">n</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div></div>",176:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype176\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> get_connections_for_tissues(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">tis1,</td></tr><tr><td class=\"PName first last\">tis2,</td></tr><tr><td class=\"PName first last\">file_name</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">Gets the points that link the AVW to CL Assumes data comes after &quot;Weld,\\n*Element, type=CONN3D2&quot; The lines look like this &quot;3, OPAL325_AVW_v6-1.14, OPAL325_Para_v6-1.15&quot;</div></div>",194:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype194\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> CurveFibersInINP(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first\">Part_Name1,</td><td></td><td class=\"last\"></td></tr><tr><td class=\"PName first\">Part_Name2,</td><td></td><td class=\"last\"></td></tr><tr><td class=\"PName first\">scale,</td><td></td><td class=\"last\"></td></tr><tr><td class=\"PName first\">inputFile,</td><td></td><td class=\"last\"></td></tr><tr><td class=\"PName first\">outputFile,</td><td></td><td class=\"last\"></td></tr><tr><td class=\"PName first\">dirVector,</td><td></td><td class=\"last\"></td></tr><tr><td class=\"PName first\">updatedP</td><td class=\"PDefaultValueSeparator\">&nbsp;=&nbsp;</td><td class=\"PDefaultValue last\"><span class=\"SHKeyword\">None</span></td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">This function takes the apical supports (or other fibers), finds the attachment points, and tries to make them a certain length</div></div>",178:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype178\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> write_part_to_inp(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">file_name,</td></tr><tr><td class=\"PName first last\">outputfile_name,</td></tr><tr><td class=\"PName first last\">part_name,</td></tr><tr><td class=\"PName first last\">data_set</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div></div>",179:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype179\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> ArcDistance(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"first\"></td><td class=\"PName last\">a,</td></tr><tr><td class=\"PSymbols first\">*</td><td class=\"PName last\">sending</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div></div>",180:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype180\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> dist(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">x1,</td></tr><tr><td class=\"PName first last\">x2,</td></tr><tr><td class=\"PName first last\">y1,</td></tr><tr><td class=\"PName first last\">y2,</td></tr><tr><td class=\"PName first last\">z1,</td></tr><tr><td class=\"PName first last\">z2</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div></div>",181:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype181\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> getFiberLength(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">fiber,</td></tr><tr><td class=\"PName first last\">inputfile</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">returns: average length of a fiber (since it may have multiple &quot;lines&quot;)</div></div>",228:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype228\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> getFiberLengths(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">inputfile,</td></tr><tr><td class=\"PName first last\">fibers</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">Calls getFiberLength n number of times, when n is the number of elements in the given fibers array</div></div>",183:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype183\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> CurvePARAFibersInINP(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">Part_Name1,</td></tr><tr><td class=\"PName first last\">Part_Name2,</td></tr><tr><td class=\"PName first last\">scale,</td></tr><tr><td class=\"PName first last\">inputFile,</td></tr><tr><td class=\"PName first last\">outputFile,</td></tr><tr><td class=\"PName first last\">dirVector,</td></tr><tr><td class=\"PName first last\">PM_Mid,</td></tr><tr><td class=\"PName first last\">connections</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">This function takes the apical supports (or other fibers), finds the attachemnt points, and tries to make them a certain length</div></div>"});