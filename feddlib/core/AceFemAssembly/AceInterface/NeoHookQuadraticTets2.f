!**************************************************************
!* AceGen    7.404 Linux (9 Feb 22)                           *
!*           Co. J. Korelc  2020           19 Apr 22 11:34:54 *
!**************************************************************
! User     : Full professional version
! Notebook : QuadraticTetsNeoHook2
! Evaluation time                 : 27 s    Mode  : Optimal
! Number of formulae              : 418     Method: Automatic
! Subroutine                      : SKR10 size: 5719
! Subroutine                      : SPP10 size: 3919
! Total size of Mathematica  code : 9638 subexpressions
! Total size of Fortran code      : 30297 bytes

 
C
      module f_routines2
        contains
!******************* S U B R O U T I N E **********************
      SUBROUTINE SKR10(v,d,ul,ul0,xl,s,p,ht,hp)
      IMPLICIT NONE
      include 'sms.h'
      INTEGER i1,i2,i206,i233,icode
      DOUBLE PRECISION v(1060),d(2),ul(3,10),ul0(3,10),xl(3
     &,10),s(30,30),p(30),ht(*),hp(*)
      icode=19
      call SMSIntPoints(icode,ngpo,gp)
      v(194)=d(2)
      v(197)=d(1)/(2d0*(1d0+v(194)))
      v(214)=v(197)/2d0
      v(195)=(2d0*v(194)*v(197))/(1d0-2d0*v(194))
      v(80)=xl(3,10)
      v(79)=xl(2,10)
      v(78)=xl(1,10)
      v(77)=xl(3,9)
      v(76)=xl(2,9)
      v(75)=xl(1,9)
      v(74)=xl(3,8)
      v(73)=xl(2,8)
      v(72)=xl(1,8)
      v(71)=xl(3,7)
      v(70)=xl(2,7)
      v(69)=xl(1,7)
      v(68)=xl(3,6)
      v(67)=xl(2,6)
      v(66)=xl(1,6)
      v(65)=xl(3,5)
      v(64)=xl(2,5)
      v(63)=xl(1,5)
      v(47)=ul(3,10)
      v(46)=ul(2,10)
      v(45)=ul(1,10)
      v(44)=ul(3,9)
      v(43)=ul(2,9)
      v(42)=ul(1,9)
      v(41)=ul(3,8)
      v(40)=ul(2,8)
      v(39)=ul(1,8)
      v(38)=ul(3,7)
      v(37)=ul(2,7)
      v(36)=ul(1,7)
      v(35)=ul(3,6)
      v(34)=ul(2,6)
      v(33)=ul(1,6)
      v(32)=ul(3,5)
      v(31)=ul(2,5)
      v(30)=ul(1,5)
      v(29)=ul(3,4)
      v(28)=ul(2,4)
      v(27)=ul(1,4)
      v(26)=ul(3,3)
      v(25)=ul(2,3)
      v(24)=ul(1,3)
      v(23)=ul(3,2)
      v(22)=ul(2,2)
      v(21)=ul(1,2)
      v(20)=ul(3,1)
      v(19)=ul(2,1)
      v(18)=ul(1,1)
      i1=ngpo
      DO i2=1,i1
       v(3)=gp(1,i2)
       v(89)=4d0*v(3)
       v(84)=(-1d0)+v(89)
       v(4)=gp(2,i2)
       v(88)=4d0*v(4)
       v(113)=-(v(77)*v(88))
       v(106)=-(v(76)*v(88))
       v(99)=-(v(75)*v(88))
       v(85)=(-1d0)+v(88)
       v(5)=gp(3,i2)
       v(90)=4d0*v(5)
       v(111)=-(v(80)*v(90))
       v(104)=-(v(79)*v(90))
       v(97)=-(v(78)*v(90))
       v(86)=(-1d0)+v(90)
       v(6)=gp(4,i2)
       v(10)=1d0-v(3)-v(4)-v(5)
       v(91)=(-4d0)*v(10)
       v(94)=-v(90)-v(91)
       v(93)=-v(88)-v(91)
       v(92)=-v(89)-v(91)
       v(87)=1d0+v(91)
       v(110)=v(87)*xl(3,4)
       v(532)=v(110)-v(74)*v(89)
       v(115)=v(113)+v(532)+v(68)*v(88)+v(71)*v(89)+v(80)*v(94)+v(86
     & )*xl(3,3)
       v(112)=v(111)+v(532)+v(65)*v(89)+v(68)*v(90)+v(77)*v(93)+v(85
     & )*xl(3,2)
       v(109)=v(110)+v(111)+v(113)+v(65)*v(88)+v(71)*v(90)+v(74)*v(92
     & )+v(84)*xl(3,1)
       v(103)=v(87)*xl(2,4)
       v(533)=v(103)-v(73)*v(89)
       v(108)=v(106)+v(533)+v(67)*v(88)+v(70)*v(89)+v(79)*v(94)+v(86
     & )*xl(2,3)
       v(105)=v(104)+v(533)+v(64)*v(89)+v(67)*v(90)+v(76)*v(93)+v(85
     & )*xl(2,2)
       v(535)=-(v(108)*v(112))+v(105)*v(115)
       v(102)=v(103)+v(104)+v(106)+v(64)*v(88)+v(70)*v(90)+v(73)*v(92
     & )+v(84)*xl(2,1)
       v(537)=-(v(105)*v(109))+v(102)*v(112)
       v(536)=v(108)*v(109)-v(102)*v(115)
       v(96)=v(87)*xl(1,4)
       v(534)=-(v(72)*v(89))+v(96)
       v(101)=v(534)+v(66)*v(88)+v(69)*v(89)+v(78)*v(94)+v(99)+v(86
     & )*xl(1,3)
       v(98)=v(534)+v(63)*v(89)+v(66)*v(90)+v(75)*v(93)+v(97)+v(85
     & )*xl(1,2)
       v(95)=v(63)*v(88)+v(69)*v(90)+v(72)*v(92)+v(96)+v(97)+v(99)+v
     & (84)*xl(1,1)
       v(126)=v(101)*v(537)+v(535)*v(95)+v(536)*v(98)
       v(117)=v(535)/v(126)
       v(118)=(v(101)*v(112)-v(115)*v(98))/v(126)
       v(119)=(-(v(101)*v(105))+v(108)*v(98))/v(126)
       v(120)=v(536)/v(126)
       v(121)=(-(v(101)*v(109))+v(115)*v(95))/v(126)
       v(122)=(v(101)*v(102)-v(108)*v(95))/v(126)
       v(123)=v(537)/v(126)
       v(124)=(-(v(112)*v(95))+v(109)*v(98))/v(126)
       v(125)=(v(105)*v(95)-v(102)*v(98))/v(126)
       v(128)=v(117)*v(84)
       v(129)=v(118)*v(84)
       v(130)=v(119)*v(84)
       v(132)=v(120)*v(85)
       v(133)=v(121)*v(85)
       v(134)=v(122)*v(85)
       v(136)=v(123)*v(86)
       v(137)=v(124)*v(86)
       v(138)=v(125)*v(86)
       v(139)=-v(117)-v(120)-v(123)
       v(140)=-v(118)-v(121)-v(124)
       v(141)=-v(119)-v(122)-v(125)
       v(143)=-(v(139)*v(87))
       v(144)=-(v(140)*v(87))
       v(145)=-(v(141)*v(87))
       v(146)=v(117)*v(88)+v(120)*v(89)
       v(147)=v(118)*v(88)+v(121)*v(89)
       v(148)=v(119)*v(88)+v(122)*v(89)
       v(149)=v(123)*v(88)+v(120)*v(90)
       v(150)=v(124)*v(88)+v(121)*v(90)
       v(151)=v(125)*v(88)+v(122)*v(90)
       v(152)=v(123)*v(89)+v(117)*v(90)
       v(153)=v(124)*v(89)+v(118)*v(90)
       v(154)=v(125)*v(89)+v(119)*v(90)
       v(155)=v(139)*v(89)-v(117)*v(91)
       v(156)=v(140)*v(89)-v(118)*v(91)
       v(157)=v(141)*v(89)-v(119)*v(91)
       v(158)=v(139)*v(88)-v(120)*v(91)
       v(159)=v(140)*v(88)-v(121)*v(91)
       v(160)=v(141)*v(88)-v(122)*v(91)
       v(161)=v(139)*v(90)-v(123)*v(91)
       v(721)=0d0
       v(722)=0d0
       v(723)=v(128)
       v(724)=0d0
       v(725)=0d0
       v(726)=v(132)
       v(727)=0d0
       v(728)=0d0
       v(729)=v(136)
       v(730)=0d0
       v(731)=0d0
       v(732)=v(143)
       v(733)=0d0
       v(734)=0d0
       v(735)=v(146)
       v(736)=0d0
       v(737)=0d0
       v(738)=v(149)
       v(739)=0d0
       v(740)=0d0
       v(741)=v(152)
       v(742)=0d0
       v(743)=0d0
       v(744)=v(155)
       v(745)=0d0
       v(746)=0d0
       v(747)=v(158)
       v(748)=0d0
       v(749)=0d0
       v(750)=v(161)
       v(691)=0d0
       v(692)=v(128)
       v(693)=0d0
       v(694)=0d0
       v(695)=v(132)
       v(696)=0d0
       v(697)=0d0
       v(698)=v(136)
       v(699)=0d0
       v(700)=0d0
       v(701)=v(143)
       v(702)=0d0
       v(703)=0d0
       v(704)=v(146)
       v(705)=0d0
       v(706)=0d0
       v(707)=v(149)
       v(708)=0d0
       v(709)=0d0
       v(710)=v(152)
       v(711)=0d0
       v(712)=0d0
       v(713)=v(155)
       v(714)=0d0
       v(715)=0d0
       v(716)=v(158)
       v(717)=0d0
       v(718)=0d0
       v(719)=v(161)
       v(720)=0d0
       v(661)=v(128)
       v(662)=0d0
       v(663)=0d0
       v(664)=v(132)
       v(665)=0d0
       v(666)=0d0
       v(667)=v(136)
       v(668)=0d0
       v(669)=0d0
       v(670)=v(143)
       v(671)=0d0
       v(672)=0d0
       v(673)=v(146)
       v(674)=0d0
       v(675)=0d0
       v(676)=v(149)
       v(677)=0d0
       v(678)=0d0
       v(679)=v(152)
       v(680)=0d0
       v(681)=0d0
       v(682)=v(155)
       v(683)=0d0
       v(684)=0d0
       v(685)=v(158)
       v(686)=0d0
       v(687)=0d0
       v(688)=v(161)
       v(689)=0d0
       v(690)=0d0
       v(162)=v(140)*v(90)-v(124)*v(91)
       v(811)=0d0
       v(812)=0d0
       v(813)=v(129)
       v(814)=0d0
       v(815)=0d0
       v(816)=v(133)
       v(817)=0d0
       v(818)=0d0
       v(819)=v(137)
       v(820)=0d0
       v(821)=0d0
       v(822)=v(144)
       v(823)=0d0
       v(824)=0d0
       v(825)=v(147)
       v(826)=0d0
       v(827)=0d0
       v(828)=v(150)
       v(829)=0d0
       v(830)=0d0
       v(831)=v(153)
       v(832)=0d0
       v(833)=0d0
       v(834)=v(156)
       v(835)=0d0
       v(836)=0d0
       v(837)=v(159)
       v(838)=0d0
       v(839)=0d0
       v(840)=v(162)
       v(781)=0d0
       v(782)=v(129)
       v(783)=0d0
       v(784)=0d0
       v(785)=v(133)
       v(786)=0d0
       v(787)=0d0
       v(788)=v(137)
       v(789)=0d0
       v(790)=0d0
       v(791)=v(144)
       v(792)=0d0
       v(793)=0d0
       v(794)=v(147)
       v(795)=0d0
       v(796)=0d0
       v(797)=v(150)
       v(798)=0d0
       v(799)=0d0
       v(800)=v(153)
       v(801)=0d0
       v(802)=0d0
       v(803)=v(156)
       v(804)=0d0
       v(805)=0d0
       v(806)=v(159)
       v(807)=0d0
       v(808)=0d0
       v(809)=v(162)
       v(810)=0d0
       v(751)=v(129)
       v(752)=0d0
       v(753)=0d0
       v(754)=v(133)
       v(755)=0d0
       v(756)=0d0
       v(757)=v(137)
       v(758)=0d0
       v(759)=0d0
       v(760)=v(144)
       v(761)=0d0
       v(762)=0d0
       v(763)=v(147)
       v(764)=0d0
       v(765)=0d0
       v(766)=v(150)
       v(767)=0d0
       v(768)=0d0
       v(769)=v(153)
       v(770)=0d0
       v(771)=0d0
       v(772)=v(156)
       v(773)=0d0
       v(774)=0d0
       v(775)=v(159)
       v(776)=0d0
       v(777)=0d0
       v(778)=v(162)
       v(779)=0d0
       v(780)=0d0
       v(163)=v(141)*v(90)-v(125)*v(91)
       v(901)=0d0
       v(902)=0d0
       v(903)=v(130)
       v(904)=0d0
       v(905)=0d0
       v(906)=v(134)
       v(907)=0d0
       v(908)=0d0
       v(909)=v(138)
       v(910)=0d0
       v(911)=0d0
       v(912)=v(145)
       v(913)=0d0
       v(914)=0d0
       v(915)=v(148)
       v(916)=0d0
       v(917)=0d0
       v(918)=v(151)
       v(919)=0d0
       v(920)=0d0
       v(921)=v(154)
       v(922)=0d0
       v(923)=0d0
       v(924)=v(157)
       v(925)=0d0
       v(926)=0d0
       v(927)=v(160)
       v(928)=0d0
       v(929)=0d0
       v(930)=v(163)
       v(871)=0d0
       v(872)=v(130)
       v(873)=0d0
       v(874)=0d0
       v(875)=v(134)
       v(876)=0d0
       v(877)=0d0
       v(878)=v(138)
       v(879)=0d0
       v(880)=0d0
       v(881)=v(145)
       v(882)=0d0
       v(883)=0d0
       v(884)=v(148)
       v(885)=0d0
       v(886)=0d0
       v(887)=v(151)
       v(888)=0d0
       v(889)=0d0
       v(890)=v(154)
       v(891)=0d0
       v(892)=0d0
       v(893)=v(157)
       v(894)=0d0
       v(895)=0d0
       v(896)=v(160)
       v(897)=0d0
       v(898)=0d0
       v(899)=v(163)
       v(900)=0d0
       v(841)=v(130)
       v(842)=0d0
       v(843)=0d0
       v(844)=v(134)
       v(845)=0d0
       v(846)=0d0
       v(847)=v(138)
       v(848)=0d0
       v(849)=0d0
       v(850)=v(145)
       v(851)=0d0
       v(852)=0d0
       v(853)=v(148)
       v(854)=0d0
       v(855)=0d0
       v(856)=v(151)
       v(857)=0d0
       v(858)=0d0
       v(859)=v(154)
       v(860)=0d0
       v(861)=0d0
       v(862)=v(157)
       v(863)=0d0
       v(864)=0d0
       v(865)=v(160)
       v(866)=0d0
       v(867)=0d0
       v(868)=v(163)
       v(869)=0d0
       v(870)=0d0
       v(177)=1d0+v(128)*v(18)+v(132)*v(21)+v(136)*v(24)+v(143)*v(27)
     & +v(146)*v(30)+v(149)*v(33)+v(152)*v(36)+v(155)*v(39)+v(158)*v
     & (42)+v(161)*v(45)
       v(178)=v(129)*v(18)+v(133)*v(21)+v(137)*v(24)+v(144)*v(27)+v
     & (147)*v(30)+v(150)*v(33)+v(153)*v(36)+v(156)*v(39)+v(159)*v(42
     & )+v(162)*v(45)
       v(179)=v(130)*v(18)+v(134)*v(21)+v(138)*v(24)+v(145)*v(27)+v
     & (148)*v(30)+v(151)*v(33)+v(154)*v(36)+v(157)*v(39)+v(160)*v(42
     & )+v(163)*v(45)
       v(180)=v(128)*v(19)+v(132)*v(22)+v(136)*v(25)+v(143)*v(28)+v
     & (146)*v(31)+v(149)*v(34)+v(152)*v(37)+v(155)*v(40)+v(158)*v(43
     & )+v(161)*v(46)
       v(181)=1d0+v(129)*v(19)+v(133)*v(22)+v(137)*v(25)+v(144)*v(28)
     & +v(147)*v(31)+v(150)*v(34)+v(153)*v(37)+v(156)*v(40)+v(159)*v
     & (43)+v(162)*v(46)
       v(182)=v(130)*v(19)+v(134)*v(22)+v(138)*v(25)+v(145)*v(28)+v
     & (148)*v(31)+v(151)*v(34)+v(154)*v(37)+v(157)*v(40)+v(160)*v(43
     & )+v(163)*v(46)
       v(183)=v(128)*v(20)+v(132)*v(23)+v(136)*v(26)+v(143)*v(29)+v
     & (146)*v(32)+v(149)*v(35)+v(152)*v(38)+v(155)*v(41)+v(158)*v(44
     & )+v(161)*v(47)
       v(184)=v(129)*v(20)+v(133)*v(23)+v(137)*v(26)+v(144)*v(29)+v
     & (147)*v(32)+v(150)*v(35)+v(153)*v(38)+v(156)*v(41)+v(159)*v(44
     & )+v(162)*v(47)
       v(185)=1d0+v(130)*v(20)+v(134)*v(23)+v(138)*v(26)+v(145)*v(29)
     & +v(148)*v(32)+v(151)*v(35)+v(154)*v(38)+v(157)*v(41)+v(160)*v
     & (44)+v(163)*v(47)
       v(186)=(v(177)*v(177))+(v(180)*v(180))+(v(183)*v(183))
       v(187)=v(177)*v(178)+v(180)*v(181)+v(183)*v(184)
       v(540)=2d0*v(187)
       v(202)=(v(187)*v(187))
       v(188)=v(177)*v(179)+v(180)*v(182)+v(183)*v(185)
       v(538)=2d0*v(188)
       v(212)=v(187)*v(538)
       v(200)=(v(188)*v(188))
       v(189)=(v(178)*v(178))+(v(181)*v(181))+(v(184)*v(184))
       v(257)=v(186)*v(189)-v(202)
       v(190)=v(178)*v(179)+v(181)*v(182)+v(184)*v(185)
       v(539)=2d0*v(190)
       v(256)=v(212)-v(186)*v(539)
       v(255)=-(v(189)*v(538))+v(187)*v(539)
       v(191)=(v(179)*v(179))+(v(182)*v(182))+(v(185)*v(185))
       v(253)=v(188)*v(539)-v(191)*v(540)
       v(252)=-(v(190)*v(190))+v(189)*v(191)
       v(254)=v(186)*v(191)-v(200)
       v(203)=-(v(189)*v(200))-v(191)*v(202)+v(190)*v(212)+v(186)*v
     & (252)
       v(251)=1d0/v(203)**2
       v(208)=-v(197)+v(195)*dlog(sqrt(v(203)))
       v(557)=v(195)-2d0*v(208)
       v(211)=v(208)/(2d0*v(203))
       v(544)=2d0*v(211)
       v(264)=v(211)*v(540)
       v(263)=v(211)*v(538)
       v(541)=2d0*(v(214)+v(211)*v(257))
       v(213)=v(211)*v(256)
       v(542)=2d0*(v(214)+v(211)*v(254))
       v(217)=v(211)*v(255)
       v(218)=v(211)*v(253)
       v(543)=2d0*(v(214)+v(211)*v(252))
       v(221)=v(184)*v(213)+v(183)*v(217)+v(185)*v(541)
       v(222)=v(181)*v(213)+v(180)*v(217)+v(182)*v(541)
       v(223)=v(178)*v(213)+v(177)*v(217)+v(179)*v(541)
       v(224)=v(185)*v(213)+v(183)*v(218)+v(184)*v(542)
       v(225)=v(182)*v(213)+v(180)*v(218)+v(181)*v(542)
       v(226)=v(179)*v(213)+v(177)*v(218)+v(178)*v(542)
       v(227)=v(185)*v(217)+v(184)*v(218)+v(183)*v(543)
       v(228)=v(182)*v(217)+v(181)*v(218)+v(180)*v(543)
       v(229)=v(179)*v(217)+v(178)*v(218)+v(177)*v(543)
       v(624)=v(130)*v(223)+v(129)*v(226)+v(128)*v(229)
       v(625)=v(130)*v(222)+v(129)*v(225)+v(128)*v(228)
       v(626)=v(130)*v(221)+v(129)*v(224)+v(128)*v(227)
       v(627)=v(134)*v(223)+v(133)*v(226)+v(132)*v(229)
       v(628)=v(134)*v(222)+v(133)*v(225)+v(132)*v(228)
       v(629)=v(134)*v(221)+v(133)*v(224)+v(132)*v(227)
       v(630)=v(138)*v(223)+v(137)*v(226)+v(136)*v(229)
       v(631)=v(138)*v(222)+v(137)*v(225)+v(136)*v(228)
       v(632)=v(138)*v(221)+v(137)*v(224)+v(136)*v(227)
       v(633)=v(145)*v(223)+v(144)*v(226)+v(143)*v(229)
       v(634)=v(145)*v(222)+v(144)*v(225)+v(143)*v(228)
       v(635)=v(145)*v(221)+v(144)*v(224)+v(143)*v(227)
       v(636)=v(148)*v(223)+v(147)*v(226)+v(146)*v(229)
       v(637)=v(148)*v(222)+v(147)*v(225)+v(146)*v(228)
       v(638)=v(148)*v(221)+v(147)*v(224)+v(146)*v(227)
       v(639)=v(151)*v(223)+v(150)*v(226)+v(149)*v(229)
       v(640)=v(151)*v(222)+v(150)*v(225)+v(149)*v(228)
       v(641)=v(151)*v(221)+v(150)*v(224)+v(149)*v(227)
       v(642)=v(154)*v(223)+v(153)*v(226)+v(152)*v(229)
       v(643)=v(154)*v(222)+v(153)*v(225)+v(152)*v(228)
       v(644)=v(154)*v(221)+v(153)*v(224)+v(152)*v(227)
       v(645)=v(157)*v(223)+v(156)*v(226)+v(155)*v(229)
       v(646)=v(157)*v(222)+v(156)*v(225)+v(155)*v(228)
       v(647)=v(157)*v(221)+v(156)*v(224)+v(155)*v(227)
       v(648)=v(160)*v(223)+v(159)*v(226)+v(158)*v(229)
       v(649)=v(160)*v(222)+v(159)*v(225)+v(158)*v(228)
       v(650)=v(160)*v(221)+v(159)*v(224)+v(158)*v(227)
       v(651)=v(163)*v(223)+v(162)*v(226)+v(161)*v(229)
       v(652)=v(163)*v(222)+v(162)*v(225)+v(161)*v(228)
       v(653)=v(163)*v(221)+v(162)*v(224)+v(161)*v(227)
       DO i206=1,30
        v(236)=v(126)*v(660+i206)
        v(237)=v(126)*v(690+i206)
        v(238)=v(126)*v(720+i206)
        v(239)=v(126)*v(750+i206)
        v(240)=v(126)*v(780+i206)
        v(241)=v(126)*v(810+i206)
        v(242)=v(126)*v(840+i206)
        v(243)=v(126)*v(870+i206)
        v(244)=v(126)*v(900+i206)
        v(245)=2d0*(v(177)*v(236)+v(180)*v(237)+v(183)*v(238))
        v(246)=v(178)*v(236)+v(181)*v(237)+v(184)*v(238)+v(177)*v(239
     &  )+v(180)*v(240)+v(183)*v(241)
        v(271)=v(246)*v(544)
        v(247)=2d0*(v(178)*v(239)+v(181)*v(240)+v(184)*v(241))
        v(248)=v(179)*v(236)+v(182)*v(237)+v(185)*v(238)+v(177)*v(242
     &  )+v(180)*v(243)+v(183)*v(244)
        v(272)=v(248)*v(544)
        v(249)=v(179)*v(239)+v(182)*v(240)+v(185)*v(241)+v(178)*v(242
     &  )+v(181)*v(243)+v(184)*v(244)
        v(265)=-(v(249)*v(544))
        v(250)=2d0*(v(179)*v(242)+v(182)*v(243)+v(185)*v(244))
        v(258)=(v(251)*(v(245)*v(252)+v(246)*v(253)+v(247)*v(254)+v
     &  (248)*v(255)+v(249)*v(256)+v(250)*v(257))*v(557))/4d0
        v(545)=2d0*(v(211)*(v(189)*v(245)+v(186)*v(247))+v(257)*v(258
     &  )-v(246)*v(264))
        v(546)=2d0*(v(211)*(v(191)*v(245)+v(186)*v(250))+v(254)*v(258
     &  )-v(248)*v(263))
        v(261)=v(211)*v(249)+v(190)*v(258)
        v(547)=2d0*(v(211)*(v(191)*v(247)+v(189)*v(250))+v(252)*v(258
     &  )+v(190)*v(265))
        v(266)=v(212)*v(258)+v(246)*v(263)+v(248)*v(264)+v(186)*v(265
     &  )-(v(211)*v(245)+v(186)*v(258))*v(539)
        v(267)=v(190)*v(271)-v(189)*v(272)-(v(211)*v(247)+v(189)*v
     &  (258))*v(538)+v(261)*v(540)
        v(268)=v(217)*v(238)+v(213)*v(241)+v(184)*v(266)+v(183)*v(267
     &  )+v(244)*v(541)+v(185)*v(545)
        v(269)=v(217)*v(237)+v(213)*v(240)+v(181)*v(266)+v(180)*v(267
     &  )+v(243)*v(541)+v(182)*v(545)
        v(270)=v(217)*v(236)+v(213)*v(239)+v(178)*v(266)+v(177)*v(267
     &  )+v(242)*v(541)+v(179)*v(545)
        v(273)=-(v(191)*v(271))+v(190)*v(272)+v(261)*v(538)-(v(211)*v
     &  (250)+v(191)*v(258))*v(540)
        v(274)=v(218)*v(238)+v(213)*v(244)+v(185)*v(266)+v(183)*v(273
     &  )+v(241)*v(542)+v(184)*v(546)
        v(275)=v(218)*v(237)+v(213)*v(243)+v(182)*v(266)+v(180)*v(273
     &  )+v(240)*v(542)+v(181)*v(546)
        v(276)=v(218)*v(236)+v(213)*v(242)+v(179)*v(266)+v(177)*v(273
     &  )+v(239)*v(542)+v(178)*v(546)
        v(277)=v(218)*v(241)+v(217)*v(244)+v(185)*v(267)+v(184)*v(273
     &  )+v(238)*v(543)+v(183)*v(547)
        v(278)=v(218)*v(240)+v(217)*v(243)+v(182)*v(267)+v(181)*v(273
     &  )+v(237)*v(543)+v(180)*v(547)
        v(279)=v(218)*v(239)+v(217)*v(242)+v(179)*v(267)+v(178)*v(273
     &  )+v(236)*v(543)+v(177)*v(547)
        v(931)=v(130)*v(270)+v(129)*v(276)+v(128)*v(279)
        v(932)=v(130)*v(269)+v(129)*v(275)+v(128)*v(278)
        v(933)=v(130)*v(268)+v(129)*v(274)+v(128)*v(277)
        v(934)=v(134)*v(270)+v(133)*v(276)+v(132)*v(279)
        v(935)=v(134)*v(269)+v(133)*v(275)+v(132)*v(278)
        v(936)=v(134)*v(268)+v(133)*v(274)+v(132)*v(277)
        v(937)=v(138)*v(270)+v(137)*v(276)+v(136)*v(279)
        v(938)=v(138)*v(269)+v(137)*v(275)+v(136)*v(278)
        v(939)=v(138)*v(268)+v(137)*v(274)+v(136)*v(277)
        v(940)=v(145)*v(270)+v(144)*v(276)+v(143)*v(279)
        v(941)=v(145)*v(269)+v(144)*v(275)+v(143)*v(278)
        v(942)=v(145)*v(268)+v(144)*v(274)+v(143)*v(277)
        v(943)=v(148)*v(270)+v(147)*v(276)+v(146)*v(279)
        v(944)=v(148)*v(269)+v(147)*v(275)+v(146)*v(278)
        v(945)=v(148)*v(268)+v(147)*v(274)+v(146)*v(277)
        v(946)=v(151)*v(270)+v(150)*v(276)+v(149)*v(279)
        v(947)=v(151)*v(269)+v(150)*v(275)+v(149)*v(278)
        v(948)=v(151)*v(268)+v(150)*v(274)+v(149)*v(277)
        v(949)=v(154)*v(270)+v(153)*v(276)+v(152)*v(279)
        v(950)=v(154)*v(269)+v(153)*v(275)+v(152)*v(278)
        v(951)=v(154)*v(268)+v(153)*v(274)+v(152)*v(277)
        v(952)=v(157)*v(270)+v(156)*v(276)+v(155)*v(279)
        v(953)=v(157)*v(269)+v(156)*v(275)+v(155)*v(278)
        v(954)=v(157)*v(268)+v(156)*v(274)+v(155)*v(277)
        v(955)=v(160)*v(270)+v(159)*v(276)+v(158)*v(279)
        v(956)=v(160)*v(269)+v(159)*v(275)+v(158)*v(278)
        v(957)=v(160)*v(268)+v(159)*v(274)+v(158)*v(277)
        v(958)=v(163)*v(270)+v(162)*v(276)+v(161)*v(279)
        v(959)=v(163)*v(269)+v(162)*v(275)+v(161)*v(278)
        v(960)=v(163)*v(268)+v(162)*v(274)+v(161)*v(277)
        p(i206)=p(i206)+v(126)*v(6)*v(623+i206)
        DO i233=1,30
         s(i206,i233)=s(i206,i233)+v(6)*v(930+i233)
        ENDDO
       ENDDO
      ENDDO
      END

!******************* S U B R O U T I N E **********************
      SUBROUTINE SPP10(v,d,ul,ul0,xl,s,p,ht,hp,sg,sg0,sxd,gpost,npost)
      IMPLICIT NONE
      include 'sms.h'
      INTEGER i283,i284,icode
      DOUBLE PRECISION v(1060),d(2),ul(3,10),ul0(3,10),xl(3
     &,10),s(30,30),p(30),ht(*),hp(*),sg(*),sg0(*),sxd(30),gpost(64
     &,21),npost(10,6)
      icode=19
      call SMSIntPoints(icode,ngpo,gp)
      v(476)=d(2)
      v(479)=d(1)/(2d0*(1d0+v(476)))
      v(500)=v(479)/2d0
      v(477)=(2d0*v(476)*v(479))/(1d0-2d0*v(476))
      v(362)=xl(3,10)
      v(361)=xl(2,10)
      v(360)=xl(1,10)
      v(359)=xl(3,9)
      v(358)=xl(2,9)
      v(357)=xl(1,9)
      v(356)=xl(3,8)
      v(355)=xl(2,8)
      v(354)=xl(1,8)
      v(353)=xl(3,7)
      v(352)=xl(2,7)
      v(351)=xl(1,7)
      v(350)=xl(3,6)
      v(349)=xl(2,6)
      v(348)=xl(1,6)
      v(347)=xl(3,5)
      v(346)=xl(2,5)
      v(345)=xl(1,5)
      v(329)=ul(3,10)
      v(328)=ul(2,10)
      v(327)=ul(1,10)
      v(326)=ul(3,9)
      v(325)=ul(2,9)
      v(324)=ul(1,9)
      v(323)=ul(3,8)
      v(322)=ul(2,8)
      v(321)=ul(1,8)
      v(320)=ul(3,7)
      v(319)=ul(2,7)
      v(318)=ul(1,7)
      v(317)=ul(3,6)
      v(316)=ul(2,6)
      v(315)=ul(1,6)
      v(314)=ul(3,5)
      v(313)=ul(2,5)
      v(312)=ul(1,5)
      v(311)=ul(3,4)
      v(310)=ul(2,4)
      v(309)=ul(1,4)
      v(308)=ul(3,3)
      v(307)=ul(2,3)
      v(306)=ul(1,3)
      v(305)=ul(3,2)
      v(304)=ul(2,2)
      v(303)=ul(1,2)
      v(302)=ul(3,1)
      v(301)=ul(2,1)
      v(300)=ul(1,1)
      i283=ngpo
      DO i284=1,i283
       v(285)=gp(1,i284)
       v(371)=4d0*v(285)
       v(366)=(-1d0)+v(371)
       v(286)=gp(2,i284)
       v(370)=4d0*v(286)
       v(395)=-(v(359)*v(370))
       v(388)=-(v(358)*v(370))
       v(381)=-(v(357)*v(370))
       v(367)=(-1d0)+v(370)
       v(287)=gp(3,i284)
       v(372)=4d0*v(287)
       v(393)=-(v(362)*v(372))
       v(386)=-(v(361)*v(372))
       v(379)=-(v(360)*v(372))
       v(368)=(-1d0)+v(372)
       v(292)=1d0-v(285)-v(286)-v(287)
       v(373)=(-4d0)*v(292)
       v(376)=-v(372)-v(373)
       v(375)=-v(370)-v(373)
       v(374)=-v(371)-v(373)
       v(369)=1d0+v(373)
       v(392)=v(369)*xl(3,4)
       v(558)=-(v(356)*v(371))+v(392)
       v(397)=v(350)*v(370)+v(353)*v(371)+v(362)*v(376)+v(395)+v(558)
     & +v(368)*xl(3,3)
       v(394)=v(347)*v(371)+v(350)*v(372)+v(359)*v(375)+v(393)+v(558)
     & +v(367)*xl(3,2)
       v(391)=v(347)*v(370)+v(353)*v(372)+v(356)*v(374)+v(392)+v(393)
     & +v(395)+v(366)*xl(3,1)
       v(385)=v(369)*xl(2,4)
       v(559)=-(v(355)*v(371))+v(385)
       v(390)=v(349)*v(370)+v(352)*v(371)+v(361)*v(376)+v(388)+v(559)
     & +v(368)*xl(2,3)
       v(387)=v(346)*v(371)+v(349)*v(372)+v(358)*v(375)+v(386)+v(559)
     & +v(367)*xl(2,2)
       v(561)=-(v(390)*v(394))+v(387)*v(397)
       v(384)=v(346)*v(370)+v(352)*v(372)+v(355)*v(374)+v(385)+v(386)
     & +v(388)+v(366)*xl(2,1)
       v(563)=-(v(387)*v(391))+v(384)*v(394)
       v(562)=v(390)*v(391)-v(384)*v(397)
       v(378)=v(369)*xl(1,4)
       v(560)=-(v(354)*v(371))+v(378)
       v(383)=v(348)*v(370)+v(351)*v(371)+v(360)*v(376)+v(381)+v(560)
     & +v(368)*xl(1,3)
       v(380)=v(345)*v(371)+v(348)*v(372)+v(357)*v(375)+v(379)+v(560)
     & +v(367)*xl(1,2)
       v(377)=v(345)*v(370)+v(351)*v(372)+v(354)*v(374)+v(378)+v(379)
     & +v(381)+v(366)*xl(1,1)
       v(408)=v(377)*v(561)+v(380)*v(562)+v(383)*v(563)
       v(399)=v(561)/v(408)
       v(400)=(v(383)*v(394)-v(380)*v(397))/v(408)
       v(401)=(-(v(383)*v(387))+v(380)*v(390))/v(408)
       v(402)=v(562)/v(408)
       v(403)=(-(v(383)*v(391))+v(377)*v(397))/v(408)
       v(404)=(v(383)*v(384)-v(377)*v(390))/v(408)
       v(405)=v(563)/v(408)
       v(406)=(v(380)*v(391)-v(377)*v(394))/v(408)
       v(407)=(-(v(380)*v(384))+v(377)*v(387))/v(408)
       v(410)=v(366)*v(399)
       v(411)=v(366)*v(400)
       v(412)=v(366)*v(401)
       v(414)=v(367)*v(402)
       v(415)=v(367)*v(403)
       v(416)=v(367)*v(404)
       v(418)=v(368)*v(405)
       v(419)=v(368)*v(406)
       v(420)=v(368)*v(407)
       v(421)=-v(399)-v(402)-v(405)
       v(422)=-v(400)-v(403)-v(406)
       v(423)=-v(401)-v(404)-v(407)
       v(425)=-(v(369)*v(421))
       v(426)=-(v(369)*v(422))
       v(427)=-(v(369)*v(423))
       v(428)=v(370)*v(399)+v(371)*v(402)
       v(429)=v(370)*v(400)+v(371)*v(403)
       v(430)=v(370)*v(401)+v(371)*v(404)
       v(431)=v(372)*v(402)+v(370)*v(405)
       v(432)=v(372)*v(403)+v(370)*v(406)
       v(433)=v(372)*v(404)+v(370)*v(407)
       v(434)=v(372)*v(399)+v(371)*v(405)
       v(435)=v(372)*v(400)+v(371)*v(406)
       v(436)=v(372)*v(401)+v(371)*v(407)
       v(437)=-(v(373)*v(399))+v(371)*v(421)
       v(438)=-(v(373)*v(400))+v(371)*v(422)
       v(439)=-(v(373)*v(401))+v(371)*v(423)
       v(440)=-(v(373)*v(402))+v(370)*v(421)
       v(441)=-(v(373)*v(403))+v(370)*v(422)
       v(442)=-(v(373)*v(404))+v(370)*v(423)
       v(443)=-(v(373)*v(405))+v(372)*v(421)
       v(444)=-(v(373)*v(406))+v(372)*v(422)
       v(445)=-(v(373)*v(407))+v(372)*v(423)
       v(447)=v(300)*v(411)+v(303)*v(415)+v(306)*v(419)+v(309)*v(426)
     & +v(312)*v(429)+v(315)*v(432)+v(318)*v(435)+v(321)*v(438)+v(324
     & )*v(441)+v(327)*v(444)
       v(448)=v(300)*v(412)+v(303)*v(416)+v(306)*v(420)+v(309)*v(427)
     & +v(312)*v(430)+v(315)*v(433)+v(318)*v(436)+v(321)*v(439)+v(324
     & )*v(442)+v(327)*v(445)
       v(449)=v(301)*v(410)+v(304)*v(414)+v(307)*v(418)+v(310)*v(425)
     & +v(313)*v(428)+v(316)*v(431)+v(319)*v(434)+v(322)*v(437)+v(325
     & )*v(440)+v(328)*v(443)
       v(451)=v(301)*v(412)+v(304)*v(416)+v(307)*v(420)+v(310)*v(427)
     & +v(313)*v(430)+v(316)*v(433)+v(319)*v(436)+v(322)*v(439)+v(325
     & )*v(442)+v(328)*v(445)
       v(452)=v(302)*v(410)+v(305)*v(414)+v(308)*v(418)+v(311)*v(425)
     & +v(314)*v(428)+v(317)*v(431)+v(320)*v(434)+v(323)*v(437)+v(326
     & )*v(440)+v(329)*v(443)
       v(453)=v(302)*v(411)+v(305)*v(415)+v(308)*v(419)+v(311)*v(426)
     & +v(314)*v(429)+v(317)*v(432)+v(320)*v(435)+v(323)*v(438)+v(326
     & )*v(441)+v(329)*v(444)
       v(455)=1d0+v(300)*v(410)+v(303)*v(414)+v(306)*v(418)+v(309)*v
     & (425)+v(312)*v(428)+v(315)*v(431)+v(318)*v(434)+v(321)*v(437)
     & +v(324)*v(440)+v(327)*v(443)
       v(456)=1d0+v(301)*v(411)+v(304)*v(415)+v(307)*v(419)+v(310)*v
     & (426)+v(313)*v(429)+v(316)*v(432)+v(319)*v(435)+v(322)*v(438)
     & +v(325)*v(441)+v(328)*v(444)
       v(457)=1d0+v(302)*v(412)+v(305)*v(416)+v(308)*v(420)+v(311)*v
     & (427)+v(314)*v(430)+v(317)*v(433)+v(320)*v(436)+v(323)*v(439)
     & +v(326)*v(442)+v(329)*v(445)
       v(517)=1d0/(v(448)*(v(449)*v(453)-v(452)*v(456))+v(447)*(v(451
     & )*v(452)-v(449)*v(457))+v(455)*(-(v(451)*v(453))+v(456)*v(457)
     & ))
       v(468)=(v(449)*v(449))+(v(452)*v(452))+(v(455)*v(455))
       v(469)=v(452)*v(453)+v(447)*v(455)+v(449)*v(456)
       v(484)=(v(469)*v(469))
       v(470)=v(449)*v(451)+v(448)*v(455)+v(452)*v(457)
       v(564)=2d0*v(470)
       v(498)=v(469)*v(564)
       v(482)=(v(470)*v(470))
       v(471)=(v(447)*v(447))+(v(453)*v(453))+(v(456)*v(456))
       v(472)=v(447)*v(448)+v(451)*v(456)+v(453)*v(457)
       v(565)=2d0*v(472)
       v(473)=(v(448)*v(448))+(v(451)*v(451))+(v(457)*v(457))
       v(566)=-(v(472)*v(472))+v(471)*v(473)
       v(485)=-(v(471)*v(482))-v(473)*v(484)+v(472)*v(498)+v(468)*v
     & (566)
       v(486)=dlog(sqrt(v(485)))
       v(489)=v(469)/2d0
       v(490)=v(470)/2d0
       v(492)=v(472)/2d0
       v(497)=(-0.5d0)*(v(479)-v(477)*v(486))/v(485)
       v(569)=2d0*((v(468)*v(471)-v(484))*v(497)+v(500))
       v(499)=v(497)*(v(498)-v(468)*v(565))
       v(568)=2d0*((v(468)*v(473)-v(482))*v(497)+v(500))
       v(503)=v(497)*(-(v(471)*v(564))+v(469)*v(565))
       v(504)=2d0*(v(470)*v(472)-v(469)*v(473))*v(497)
       v(567)=2d0*(v(500)+v(497)*v(566))
       v(507)=v(448)*v(503)+v(447)*v(504)+v(455)*v(567)
       v(508)=v(448)*v(499)+v(455)*v(504)+v(447)*v(568)
       v(509)=v(447)*v(499)+v(455)*v(503)+v(448)*v(569)
       v(510)=v(451)*v(503)+v(456)*v(504)+v(449)*v(567)
       v(511)=v(451)*v(499)+v(449)*v(504)+v(456)*v(568)
       v(512)=v(456)*v(499)+v(449)*v(503)+v(451)*v(569)
       v(516)=(v(455)*v(507)+v(447)*v(508)+v(448)*v(509))*v(517)
       v(518)=(v(449)*v(507)+v(456)*v(508)+v(451)*v(509))*v(517)
       v(519)=(v(452)*v(507)+v(453)*v(508)+v(457)*v(509))*v(517)
       v(521)=(v(449)*v(510)+v(456)*v(511)+v(451)*v(512))*v(517)
       v(522)=(v(452)*v(510)+v(453)*v(511)+v(457)*v(512))*v(517)
       v(525)=v(517)*(v(452)*(v(457)*v(503)+v(453)*v(504)+v(452)*v
     & (567))+v(453)*(v(457)*v(499)+v(452)*v(504)+v(453)*v(568))+v
     & (457)*(v(453)*v(499)+v(452)*v(503)+v(457)*v(569)))
       v(527)=(-v(516)-v(521)-v(525))/3d0
       gpost(i284,1)=v(516)
       gpost(i284,2)=v(518)
       gpost(i284,3)=v(519)
       gpost(i284,4)=v(518)
       gpost(i284,5)=v(521)
       gpost(i284,6)=v(522)
       gpost(i284,7)=v(519)
       gpost(i284,8)=v(522)
       gpost(i284,9)=v(525)
       gpost(i284,10)=((-1d0)+v(468))/2d0
       gpost(i284,11)=v(489)
       gpost(i284,12)=v(490)
       gpost(i284,13)=v(489)
       gpost(i284,14)=((-1d0)+v(471))/2d0
       gpost(i284,15)=v(492)
       gpost(i284,16)=v(490)
       gpost(i284,17)=v(492)
       gpost(i284,18)=((-1d0)+v(473))/2d0
       gpost(i284,19)=sqrt(0.15d1*(2d0*(v(518)*v(518))+2d0*(v(519)*v
     & (519))+2d0*(v(522)*v(522))+(v(516)+v(527))**2+(v(521)+v(527)
     & )**2+(v(525)+v(527))**2))
       gpost(i284,20)=v(479)*(((-3d0)+v(468)+v(471)+v(473))/2d0-v(486
     & ))+(v(477)*(v(486)*v(486)))/2d0
       gpost(i284,21)=gp(4,i284)*v(408)
      ENDDO
      npost(1,1)=v(300)
      npost(2,1)=v(303)
      npost(3,1)=v(306)
      npost(4,1)=v(309)
      npost(5,1)=v(312)
      npost(6,1)=v(315)
      npost(7,1)=v(318)
      npost(8,1)=v(321)
      npost(9,1)=v(324)
      npost(10,1)=v(327)
      npost(1,2)=v(301)
      npost(2,2)=v(304)
      npost(3,2)=v(307)
      npost(4,2)=v(310)
      npost(5,2)=v(313)
      npost(6,2)=v(316)
      npost(7,2)=v(319)
      npost(8,2)=v(322)
      npost(9,2)=v(325)
      npost(10,2)=v(328)
      npost(1,3)=v(302)
      npost(2,3)=v(305)
      npost(3,3)=v(308)
      npost(4,3)=v(311)
      npost(5,3)=v(314)
      npost(6,3)=v(317)
      npost(7,3)=v(320)
      npost(8,3)=v(323)
      npost(9,3)=v(326)
      npost(10,3)=v(329)
      npost(1,4)=v(300)
      npost(2,4)=v(303)
      npost(3,4)=v(306)
      npost(4,4)=v(309)
      npost(5,4)=v(312)
      npost(6,4)=v(315)
      npost(7,4)=v(318)
      npost(8,4)=v(321)
      npost(9,4)=v(324)
      npost(10,4)=v(327)
      npost(1,5)=v(301)
      npost(2,5)=v(304)
      npost(3,5)=v(307)
      npost(4,5)=v(310)
      npost(5,5)=v(313)
      npost(6,5)=v(316)
      npost(7,5)=v(319)
      npost(8,5)=v(322)
      npost(9,5)=v(325)
      npost(10,5)=v(328)
      npost(1,6)=v(302)
      npost(2,6)=v(305)
      npost(3,6)=v(308)
      npost(4,6)=v(311)
      npost(5,6)=v(314)
      npost(6,6)=v(317)
      npost(7,6)=v(320)
      npost(8,6)=v(323)
      npost(9,6)=v(326)
      npost(10,6)=v(329)
      END
      end module f_routines2
