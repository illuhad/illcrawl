/*
 * This file is part of illcrawl, a reconstruction engine for data from
 * the illustris simulation.
 *
 * Copyright (C) 2017  Aksel Alpay
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "chandra.hpp"

namespace illcrawl {
namespace chandra {

static constexpr device_scalar arf_data [] = {
  7.977484415e-07f,
  3.577559582e-08f,
  7.116806255e-09f,
  5.609000198e-08f,
  3.907619828e-07f,
  5.300075259e-07f,
  2.869025138e-06f,
  3.895911141e-05f,
  0.0001386822405f,
  0.0004844235955f,
  0.00166438054293f,
  0.00617359718308f,
  0.0182332973927f,
  0.0461972691119f,
  0.103154294193f,
  0.209841325879f,
  0.388258457184f,
  0.0645755529404f,
  6.126869266e-05f,
  0.0005555900862f,
  0.00324052711949f,
  0.0082097845152f,
  0.0167899709195f,
  0.0287541393191f,
  0.0432579480112f,
  0.0647673383355f,
  0.101018205285f,
  0.154965266585f,
  0.185234710574f,
  0.283606261015f,
  0.441627293825f,
  0.653877675533f,
  0.92331391573f,
  1.2696349621f,
  1.7244797945f,
  2.29818964f,
  2.9920933247f,
  3.8804073334f,
  4.9872689247f,
  6.2717733383f,
  0.152047619224f,
  0.0178924240172f,
  0.246096879244f,
  0.284224927425f,
  0.719184696674f,
  0.995271086693f,
  1.3337602615f,
  1.6553449631f,
  2.0600984097f,
  2.6413094997f,
  3.3122520447f,
  4.1190671921f,
  5.0709457397f,
  6.1645860672f,
  7.4131331444f,
  3.4563789368f,
  5.8319115639f,
  7.4242458344f,
  8.278427124f,
  9.6656742096f,
  11.5478658676f,
  13.690117836f,
  16.0502243042f,
  18.6791095734f,
  21.5936012268f,
  24.6866436005f,
  27.9324417114f,
  31.416727066f,
  35.1219367981f,
  38.8272705078f,
  42.8372955322f,
  47.0213165283f,
  51.278755188f,
  55.6355209351f,
  59.9581260681f,
  64.3754043579f,
  68.9152832031f,
  73.692817688f,
  78.5166854858f,
  83.5099411011f,
  88.7097244263f,
  94.1064758301f,
  99.8207778931f,
  105.329429627f,
  110.853134155f,
  116.362045288f,
  122.092536926f,
  128.03918457f,
  134.075668335f,
  140.09954834f,
  146.153427124f,
  152.134094238f,
  158.287475586f,
  164.4090271f,
  170.75227356f,
  177.398590088f,
  184.117706299f,
  190.638046265f,
  197.015899658f,
  203.180648804f,
  209.113067627f,
  215.074066162f,
  221.16305542f,
  227.393600464f,
  233.678222656f,
  239.736206055f,
  245.85093689f,
  251.902664185f,
  257.809631348f,
  263.819091797f,
  269.934570312f,
  275.999542236f,
  282.15145874f,
  286.069793701f,
  291.802917481f,
  297.536682129f,
  303.131378174f,
  308.65701294f,
  314.267608643f,
  320.051483154f,
  325.746765137f,
  331.234222412f,
  336.563049316f,
  341.774505615f,
  346.760742188f,
  351.784362793f,
  356.796844482f,
  361.802398682f,
  366.714813232f,
  371.592254639f,
  376.517669678f,
  381.452178955f,
  386.085693359f,
  390.310241699f,
  394.530029297f,
  398.404449463f,
  392.968231201f,
  324.380004883f,
  338.504364014f,
  354.842285156f,
  348.468017578f,
  356.879119873f,
  369.678070068f,
  367.577453613f,
  369.7762146f,
  374.507659912f,
  382.825775147f,
  384.843505859f,
  386.305725098f,
  388.681762695f,
  393.779815674f,
  397.841766357f,
  401.682647705f,
  405.15737915f,
  407.670440674f,
  410.54208374f,
  414.806182861f,
  418.09387207f,
  421.209167481f,
  423.382232666f,
  425.94909668f,
  428.558990478f,
  430.912902832f,
  259.406494141f,
  245.544296265f,
  263.595581055f,
  292.503570557f,
  301.572937012f,
  305.117095947f,
  298.834381103f,
  313.851318359f,
  313.102722168f,
  314.753448486f,
  319.294067383f,
  322.004089356f,
  322.808288574f,
  329.904113769f,
  333.369689941f,
  332.593841553f,
  330.690338135f,
  325.851898193f,
  320.97277832f,
  314.443603516f,
  294.266998291f,
  261.237457275f,
  216.665420532f,
  199.420578003f,
  206.989318848f,
  209.709503174f,
  209.123397827f,
  199.370254517f,
  195.884460449f,
  185.989822388f,
  181.410934448f,
  190.47946167f,
  194.505111694f,
  198.810653686f,
  204.194000244f,
  211.041885376f,
  216.608016968f,
  220.664611816f,
  223.120910644f,
  222.803833008f,
  226.528640747f,
  233.001907349f,
  238.038314819f,
  240.977767944f,
  242.615859985f,
  243.485549927f,
  244.797531128f,
  249.69581604f,
  253.281570435f,
  257.232940674f,
  260.342712402f,
  261.393188477f,
  261.693847656f,
  263.21282959f,
  266.44519043f,
  269.090759277f,
  270.993438721f,
  272.756958008f,
  274.355712891f,
  274.704376221f,
  274.417144775f,
  274.722076416f,
  274.807434082f,
  275.356262207f,
  275.09298706f,
  272.13760376f,
  260.043334961f,
  247.265914917f,
  263.866851807f,
  268.498687744f,
  271.353607178f,
  275.040344238f,
  276.961791992f,
  280.420715332f,
  283.134094238f,
  285.412994385f,
  286.543792725f,
  287.974151611f,
  289.582366943f,
  291.242401123f,
  293.521057129f,
  295.054260254f,
  296.235687256f,
  297.785217285f,
  299.443603516f,
  300.878326416f,
  302.523162842f,
  304.207458496f,
  305.463409424f,
  306.411193848f,
  307.11138916f,
  307.904022217f,
  308.81060791f,
  310.179718018f,
  311.762573242f,
  313.122894287f,
  314.101318359f,
  314.399841309f,
  313.921691894f,
  312.289672852f,
  307.236511231f,
  309.001770019f,
  312.512451172f,
  314.227203369f,
  315.845886231f,
  316.911346435f,
  317.941101074f,
  319.38293457f,
  321.085540772f,
  322.512054443f,
  323.542510986f,
  324.873199463f,
  325.916900635f,
  326.815979004f,
  327.812286377f,
  328.655426025f,
  329.384552002f,
  330.144226074f,
  331.026123047f,
  331.917755127f,
  332.911743164f,
  333.497283935f,
  333.781738281f,
  333.756622315f,
  333.042999268f,
  331.259307861f,
  329.022033691f,
  329.962097168f,
  332.116333008f,
  333.227600098f,
  335.278869629f,
  336.582489014f,
  337.369934082f,
  338.469909668f,
  339.614349365f,
  340.719268799f,
  341.681854248f,
  342.646209717f,
  343.266174316f,
  343.816650391f,
  344.617858887f,
  345.644683838f,
  346.684570312f,
  347.777587891f,
  348.872619629f,
  349.488525391f,
  349.998168945f,
  350.650909424f,
  351.458068848f,
  352.205474853f,
  352.556030273f,
  352.906585693f,
  353.457855225f,
  354.060119629f,
  354.675079346f,
  355.306365967f,
  355.937286377f,
  356.558013916f,
  357.179382324f,
  357.702026367f,
  358.189819336f,
  358.695159912f,
  359.227752685f,
  359.755767822f,
  360.167144775f,
  360.578674316f,
  361.080444336f,
  361.622558594f,
  362.118530273f,
  362.503326416f,
  362.867523193f,
  363.179382324f,
  363.486999512f,
  363.77923584f,
  364.112213135f,
  364.438293457f,
  364.747772217f,
  365.055847168f,
  365.218780518f,
  365.374023438f,
  365.56564331f,
  365.781463623f,
  366.003265381f,
  366.24307251f,
  366.482849121f,
  366.619537353f,
  366.74508667f,
  366.977233887f,
  367.288513184f,
  367.554779053f,
  367.647979736f,
  367.740997315f,
  367.896606445f,
  368.062469482f,
  368.201599121f,
  368.314941406f,
  368.466186523f,
  368.816772461f,
  369.167510986f,
  369.522247315f,
  369.878082275f,
  370.214691162f,
  370.529144287f,
  370.809967041f,
  371.148010254f,
  371.48626709f,
  371.450073242f,
  371.297302246f,
  371.206237793f,
  371.204284668f,
  371.197418213f,
  371.126251221f,
  371.051177978f,
  370.992401123f,
  370.944641113f,
  370.858215332f,
  370.706329346f,
  370.555267334f,
  370.448791504f,
  370.342041016f,
  370.2237854f,
  370.099792481f,
  369.961975098f,
  369.795684815f,
  369.629119873f,
  369.452667236f,
  369.275787353f,
  369.102722168f,
  368.931884766f,
  368.734680176f,
  368.470153809f,
  368.205200195f,
  367.996276856f,
  367.75100708f,
  367.504455566f,
  367.294372559f,
  367.029022217f,
  366.582977295f,
  366.136688232f,
  365.874603272f,
  365.636322022f,
  365.294067383f,
  364.862030029f,
  364.452819824f,
  364.143005371f,
  363.83303833f,
  363.375427246f,
  362.889404297f,
  362.469451904f,
  362.118103027f,
  361.756347656f,
  361.332427978f,
  360.908325195f,
  360.495788574f,
  360.086181641f,
  359.584594727f,
  358.968292236f,
  358.359619141f,
  357.824768066f,
  357.289703369f,
  356.707550049f,
  356.109405518f,
  355.493835449f,
  354.95223999f,
  354.399688721f,
  353.623840332f,
  352.848083496f,
  352.233673096f,
  351.680358887f,
  351.069702148f,
  350.36026001f,
  349.651123047f,
  348.998077393f,
  348.345184326f,
  347.639221191f,
  346.905517578f,
  346.185913086f,
  345.498138428f,
  344.810455322f,
  343.969024658f,
  343.121124268f,
  342.405853272f,
  341.775390625f,
  341.082733154f,
  340.214050293f,
  339.345581055f,
  338.612335205f,
  337.892211914f,
  337.081939697f,
  336.135009766f,
  335.170410156f,
  334.253082275f,
  333.336395264f,
  332.41506958f,
  331.493530273f,
  330.537719727f,
  329.536315918f,
  328.52810669f,
  327.483734131f,
  326.440155029f,
  325.416503906f,
  324.397796631f,
  323.361907959f,
  322.306854248f,
  321.246551514f,
  320.144042969f,
  319.042358398f,
  317.988311768f,
  316.948638916f,
  315.898254394f,
  314.833282471f,
  313.7605896f,
  312.588317871f,
  311.417114258f,
  310.29788208f,
  309.189300537f,
  308.008850098f,
  306.829345703f,
  305.648132324f,
  304.374847412f,
  303.103149414f,
  301.86428833f,
  300.641540527f,
  299.403656006f,
  298.134399414f,
  296.866729736f,
  295.601013184f,
  294.336883545f,
  293.053192139f,
  291.759094238f,
  290.46572876f,
  289.171661377f,
  287.879241943f,
  286.484405518f,
  285.084686279f,
  283.748199463f,
  282.455688477f,
  281.173675537f,
  279.920501709f,
  278.668914795f,
  277.321777344f,
  275.924743652f,
  274.610534668f,
  273.365325928f,
  272.086853027f,
  270.667327881f,
  269.25f,
  268.031860352f,
  266.850708008f,
  265.588745117f,
  264.250274658f,
  262.92276001f,
  261.646789551f,
  260.372924805f,
  259.108337402f,
  257.847595215f,
  256.60470581f,
  255.3802948f,
  254.164291382f,
  253.005203247f,
  251.847885132f,
  250.685699463f,
  249.523208618f,
  248.404281616f,
  247.34211731f,
  246.232299805f,
  245.069519043f,
  243.908920288f,
  242.801757812f,
  241.717407227f,
  240.592376709f,
  239.394424439f,
  238.2006073f,
  237.143035889f,
  236.08732605f,
  234.998123169f,
  233.89302063f,
  232.804504394f,
  231.749191284f,
  230.695755005f,
  229.63482666f,
  228.575500488f,
  227.695480347f,
  226.925842285f,
  226.046768189f,
  224.871292114f,
  223.698043823f,
  222.646972656f,
  221.60798645f,
  220.540512085f,
  219.448501587f,
  218.381942749f,
  217.397537231f,
  216.415145874f,
  215.432556152f,
  214.451660156f,
  213.448440552f,
  212.425598144f,
  211.412124634f,
  210.433868408f,
  209.457611084f,
  208.482269287f,
  207.508743286f,
  206.493774414f,
  205.434188843f,
  204.395584106f,
  203.480987549f,
  202.568283081f,
  201.53225708f,
  200.464080811f,
  199.464706421f,
  198.553314209f,
  197.630584717f,
  196.570892334f,
  195.493026733f,
  194.433303833f,
  193.383758545f,
  192.354431152f,
  191.355117798f,
  190.357727051f,
  189.354705811f,
  188.353973389f,
  187.331161499f,
  186.299942017f,
  185.235076904f,
  184.104095459f,
  182.975814819f,
  181.906341553f,
  180.839675903f,
  179.842422485f,
  178.883926392f,
  177.87727356f,
  176.754684448f,
  175.634948731f,
  174.529998779f,
  173.428436279f,
  172.295211792f,
  171.142181397f,
  170.014205933f,
  168.952529907f,
  167.891998291f,
  166.702224731f,
  165.501602173f,
  164.358200073f,
  163.261047363f,
  162.144744873f,
  160.978561401f,
  159.815628052f,
  158.68637085f,
  157.565200806f,
  156.427719116f,
  155.265029907f,
  154.114334106f,
  153.020492554f,
  151.929672241f,
  150.750427246f,
  149.553359985f,
  148.394515991f,
  147.279953003f,
  146.162475586f,
  145.001159668f,
  143.839614868f,
  142.722076416f,
  141.623855591f,
  140.482727051f,
  139.280700684f,
  138.087432861f,
  136.972351074f,
  135.873229981f,
  134.784713745f,
  133.702178955f,
  132.610717773f,
  131.502227783f,
  130.396377564f,
  129.263748169f,
  128.134552002f,
  127.051849365f,
  125.985046387f,
  124.918533325f,
  123.852790833f,
  122.789321899f,
  121.75353241f,
  120.721595764f,
  119.710426331f,
  118.710273743f,
  117.685852051f,
  116.594413757f,
  115.506477356f,
  114.488105774f,
  113.477813721f,
  112.485359192f,
  111.506462097f,
  110.525650024f,
  109.531463623f,
  108.540405273f,
  107.568435669f,
  106.6015625f,
  105.634536743f,
  104.667816162f,
  103.725326538f,
  102.876945496f,
  102.031173706f,
  101.067970276f,
  100.084976196f,
  99.2031097412f,
  98.4253692627f,
  97.6350402832f,
  96.7571029663f,
  95.882019043f,
  95.0498123169f,
  94.2306747437f,
  93.3817520142f,
  92.4953079224f,
  91.6173171997f,
  90.7939605713f,
  89.9733352661f,
  89.147354126f,
  88.321472168f,
  87.4895248413f,
  86.6470794678f,
  85.8106460571f,
  85.0362625122f,
  84.264503479f,
  83.4385147095f,
  82.5912475586f,
  81.7854003906f,
  81.0526504517f,
  80.3221206665f,
  79.5414123535f,
  78.763381958f,
  77.9777450562f,
  77.1894073486f,
  76.4206314087f,
  75.692199707f,
  74.9662475586f,
  74.2013244629f,
  73.43724823f,
  72.7637023926f,
  72.148475647f,
  71.4688873291f,
  70.6045837402f,
  69.7434005737f,
  69.0385818481f,
  68.3508300781f,
  67.6562957764f,
  66.9571533203f,
  66.2729797363f,
  65.6367416382f,
  65.0027236938f,
  64.338973999f,
  63.6726875305f,
  63.0120201111f,
  62.3567581177f,
  61.7100982666f,
  61.0963172913f,
  60.4846992493f,
  59.8595848083f,
  59.2333259583f,
  58.5984268188f,
  57.9536857605f,
  57.3209838867f,
  56.7589111328f,
  56.1987686157f,
  55.6294784546f,
  55.0590057373f,
  54.4829330444f,
  53.898651123f,
  53.3219299316f,
  52.8111839294f,
  52.3022422791f,
  51.7807273865f,
  51.2557258606f,
  50.7381477356f,
  50.2313995361f,
  49.7332344055f,
  49.2611083984f,
  48.7905273438f,
  48.3033714294f,
  47.8094711304f,
  47.3394966125f,
  46.9152793884f,
  46.4924354553f,
  46.0701217651f,
  45.6491699219f,
  45.2604560852f,
  44.8905296326f,
  44.5121574402f,
  44.1114578247f,
  43.7120628357f,
  43.3547515869f,
  43.0012321472f,
  42.6618080139f,
  42.3324317932f,
  42.009563446f,
  41.7047233582f,
  41.4008712769f,
  41.0999336243f,
  40.8002166748f,
  40.5032348633f,
  40.2087097168f,
  39.9204063416f,
  39.6542816162f,
  39.3890190125f,
  39.1222457886f,
  38.8563346863f,
  38.6039466858f,
  38.3649368286f,
  38.1284675598f,
  37.9028091431f,
  37.6777992249f,
  37.4316482544f,
  37.1808891296f,
  36.9428024292f,
  36.7196464539f,
  36.4963989258f,
  36.2674598694f,
  36.0391998291f,
  35.8170318604f,
  35.5972328186f,
  35.3871040344f,
  35.1887321472f,
  34.9906044006f,
  34.7885055542f,
  34.5869560242f,
  34.393989563f,
  34.2048149109f,
  34.0166320801f,
  33.8298339844f,
  33.643611908f,
  33.4632720947f,
  33.2834320068f,
  33.1038742065f,
  32.9247131348f,
  32.7474098206f,
  32.5735664368f,
  32.4002113342f,
  32.220085144f,
  32.0402526855f,
  31.8798789978f,
  31.7315578461f,
  31.5804805756f,
  31.4213600159f,
  31.2626628876f,
  31.100566864f,
  30.9385738373f,
  30.7796478271f,
  30.6230621338f,
  30.4690647125f,
  30.3228816986f,
  30.177066803f,
  30.0250968933f,
  29.8725891113f,
  29.7209453583f,
  29.5700798035f,
  29.4202613831f,
  29.2738418579f,
  29.127784729f,
  28.9762458801f,
  28.8238983154f,
  28.6764297485f,
  28.5371932983f,
  28.3986320496f,
  28.2626399994f,
  28.1269683838f,
  27.9678268433f,
  27.8026218414f,
  27.6526546478f,
  27.5221843719f,
  27.3903751373f,
  27.2417259216f,
  27.0934963226f,
  26.9252796173f,
  26.7503871918f,
  26.5807876587f,
  26.4191455841f,
  26.259141922f,
  26.1260509491f,
  25.9933223724f,
  25.8668422699f,
  25.7432975769f,
  25.6071357727f,
  25.4467468262f,
  25.2868614197f,
  25.1361122131f,
  24.9858551025f,
  24.8528327942f,
  24.7293014526f,
  24.5989093781f,
  24.451997757f,
  24.3054428101f,
  24.1642379761f,
  24.0237007141f,
  23.8903560638f,
  23.7618675232f,
  23.6254367828f,
  23.4648990631f,
  23.3048820496f,
  23.1840381622f,
  23.070734024f,
  22.9377002716f,
  22.7890586853f,
  22.6429977417f,
  22.5055541992f,
  22.3685207367f,
  22.2317047119f,
  22.095287323f,
  21.956577301f,
  21.8157157898f,
  21.6758384705f,
  21.539270401f,
  21.4031448364f,
  21.2586097717f,
  21.1125354767f,
  20.9678173065f,
  20.824596405f,
  20.6810646057f,
  20.5317554474f,
  20.3829860687f,
  20.2404136658f,
  20.100069046f,
  19.9556827545f,
  19.8054771423f,
  19.6551036835f,
  19.4955234528f,
  19.3365478516f,
  19.1835384369f,
  19.033159256f,
  18.8775367737f,
  18.712677002f,
  18.5484294891f,
  18.383398056f,
  18.21900177f,
  18.061258316f,
  17.906999588f,
  17.7467823029f,
  17.5742835999f,
  17.4024372101f,
  17.2350101471f,
  17.0683040619f,
  16.8870296478f,
  16.6975231171f,
  16.5112094879f,
  16.3319244385f,
  16.1533489227f,
  15.9580354691f,
  15.7622699738f,
  15.567984581f,
  15.3749990463f,
  15.1803150177f,
  14.9783010483f,
  14.7771654129f,
  14.5751924515f,
  14.3738899231f,
  14.1710624695f,
  13.9670715332f,
  13.7633609772f,
  13.5578117371f,
  13.3531866074f,
  13.1512517929f,
  12.9505405426f,
  12.7525968552f,
  12.5574417114f,
  12.3616552353f,
  12.1576547623f,
  11.9545660019f,
  11.7588376999f,
  11.5656375885f,
  11.3711385727f,
  11.1747550964f,
  10.9785385132f,
  10.7783966064f,
  10.5799150467f,
  10.3823442459f,
  10.185590744f,
  9.9858551025f,
  9.7812700272f,
  9.5778589249f,
  9.3817071915f,
  9.1863565445f,
  8.9914808273f,
  8.7972984314f,
  8.6058530807f,
  8.4187669754f,
  8.2324542999f,
  8.0457601547f,
  7.8598847389f,
  7.6723928452f,
  7.4844779968f,
  7.3027715683f,
  7.1340179443f,
  6.9660134315f,
  6.8049945831f,
  6.6449637413f,
  6.4807186127f,
  6.3140740395f,
  6.1522812843f,
  6.0027332306f,
  5.8538637161f,
  5.694633007f,
  5.5350513458f,
  5.371617794f,
  5.2053804398f,
  5.0477523804f,
  4.9192481041f,
  4.7919335365f,
  4.6607646942f,
  4.5294771194f,
  4.3957390785f,
  4.2597646713f,
  4.1245055199f,
  3.9904379845f,
  3.8570570946f,
  3.7335135937f,
  3.6125164032f,
  3.4905936718f,
  3.3675954342f,
  3.2463490963f,
  3.1341757774f,
  3.0224995613f,
  2.9081742764f,
  2.7934684753f,
  2.6847822666f,
  2.5839819908f,
  2.4839842319f,
  2.3879451752f,
  2.292342186f,
  2.2075212002f,
  2.1269214153f,
  2.0492782593f,
  1.9762071371f,
  1.9038044214f,
  1.8411972523f,
  1.7788728476f,
  1.7209668159f,
  1.6652337313f,
  1.6103476286f,
  1.556878686f,
  1.5036540031f,
  1.4625056982f,
  1.4217047691f,
  1.3804521561f,
  1.339019537f,
  1.2985463142f,
  1.2601416111f,
  1.2219099998f,
  1.1866141558f,
  1.1517899036f,
  1.1177762747f,
  1.0843440294f,
  1.0516204834f,
  1.0207937956f,
  0.990097761154f,
  0.960764706135f,
  0.931646764278f,
  0.903493225574f,
  0.876161396503f,
  0.848788499832f,
  0.820886850357f,
  0.793109357357f,
  0.765801787376f,
  0.738727629185f
};

math::scalar arf::arf_min_energy()
{
  return 0.1053223;
}

math::scalar arf::arf_bin_width()
{
  return 0.010644;
}

math::scalar arf::arf_max_energy()
{
  return arf_min_energy() + 1024 * arf_bin_width();
}

std::size_t arf::get_num_arf_bins()
{
  return 1024;
}

arf::arf(const qcl::device_context_ptr& ctx)
  : tabulated_function{ctx,
                       arf_data,        // data begin
                       arf_data + 1024, // data end
                       0.105322271585f, // x-range begins at 0.105 keV
                       0.010644f}       // dx = 0.0106 keV
{}


}
}
