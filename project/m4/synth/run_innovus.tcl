# ================================================================
# run_innovus.tcl   Cadence Innovus 17.1
# Design : top_ti   SAED14nm RVT 500 MHz  — 32×32 systolic array
#
# Changes vs. single-tile run:
#   - Netlist/SDC: point to 32×32 Genus outputs
#   - Floorplan  : 110×110 µm → 2620×2620 µm die
#                  (2600×2600 µm core, 10 µm boundary margin)
#                  Adjust after checking area_32x32.rpt if actual
#                  cell area differs significantly from estimate.
#   - Power      : add horizontal M8 stripes to complement vertical
#                  M7 stripes — prevents IR drop across the wide die
#   - Placement  : density 0.65 → 0.60 to give routing headroom for
#                  the 144-bit weight broadcast net (1024 loads)
#   - CTS        : target_skew 0.04 → 0.06 ns — the 32×32 clock tree
#                  drives 1024× more flip-flops; 40 ps is unreachable
#   - Outputs    : all filenames → top_ti_32x32_*
#   - Fixed      : duplicate script body removed
# ================================================================

set DESIGN   top_ti
set BASE     /u/amoghvt/Desktop/HW-AI
set PDK      /pkgs/synopsys/libs/saed14nm
set PNR      $BASE/pnr_32x32

file mkdir $PNR

# ----------------------------------------------------------------
# 1. Read design
# ----------------------------------------------------------------
read_mmmc $BASE/pnr/mmmc.tcl

read_physical -ndm [list \
    $PDK/stdcell_rvt/ndm/tech_ndm/1p9m_tech.ndm \
    $PDK/stdcell_rvt/ndm/saed14rvt_frame_only.ndm \
]

# Point to the 32×32 Genus netlist (not the single-tile one).
read_netlist $BASE/reports/top_ti_32x32_netlist.v -top $DESIGN

init_design

set_analysis_view -setup { setup_view } -hold { hold_view }

puts "INFO: Design initialized."

# ----------------------------------------------------------------
# 2. Floorplan
#
# Die 2620×2620 µm, core 2600×2600 µm, 10 µm boundary margin.
# Starting point from task spec; check area_32x32.rpt after Genus:
#   - If reported cell area < 3 Mµm², shrink to 1800×1800 µm.
#   - If reported cell area > 5 Mµm², grow to 3000×3000 µm.
# Target utilization: ~0.60 (leave 40% for routing the weight bus).
# ----------------------------------------------------------------
floorplan \
    -site      unit \
    -die_size  2620 2620 10 10 10 10 \
    -core_size 2600 2600 10 10

snapFPlanIO -toRows

# Distribute ports across all four sides — a 32×32 design has too
# many ports (AXI-Lite 13-bit + AXI-Stream 512-bit) to pack on LEFT.
editPin -fixOverlap 1 -unit TRACK -spreadType CENTER \
    -layer 3 -spreadDir clockwise -pin {s_axil_*} -side LEFT
editPin -fixOverlap 1 -unit TRACK -spreadType CENTER \
    -layer 3 -spreadDir clockwise -pin {s_axis_*} -side BOTTOM
editPin -fixOverlap 1 -unit TRACK -spreadType CENTER \
    -layer 3 -spreadDir clockwise -pin {clk rst_n} -side RIGHT

puts "INFO: Floorplan done."

# ----------------------------------------------------------------
# 3. Power planning
#
# Ring: M9 (H) / M8 (V) — same as single-tile, width kept at 1.0.
# Stripes: added M8 horizontal to complement M7 vertical so the
#   power grid forms a 2D mesh across the 2600 µm die.  A 1D stripe
#   pattern would leave half the rows under-supplied.
# Stripe pitch: 8 µm → 6 µm (denser) because the 32×32 array has
#   higher current density from 1024 simultaneous MAC units.
# ----------------------------------------------------------------
globalNetConnect VDD -type pgpin -pin VDD  -inst * -verbose
globalNetConnect VSS -type pgpin -pin VSS  -inst * -verbose
globalNetConnect VDD -type pgpin -pin VDDR -inst * -verbose

addRing \
    -nets         { VDD VSS } \
    -type         core_rings \
    -layer_top    M9 \
    -layer_bottom M9 \
    -layer_left   M8 \
    -layer_right  M8 \
    -width 1.0 -spacing 0.5 -offset 0.5

# Vertical stripes on M7 (same layer as single-tile run)
addStripe \
    -nets                { VDD VSS } \
    -layer               M7 \
    -direction           vertical \
    -width               0.6 \
    -spacing             0.4 \
    -set_to_set_distance 6.0 \
    -start_from          left \
    -start_offset        6.0 \
    -stop_offset         6.0

# Horizontal stripes on M8 — new for 32×32, forms 2D mesh with M7
addStripe \
    -nets                { VDD VSS } \
    -layer               M8 \
    -direction           horizontal \
    -width               0.6 \
    -spacing             0.4 \
    -set_to_set_distance 6.0 \
    -start_from          bottom \
    -start_offset        6.0 \
    -stop_offset         6.0

sroute -connect { corePin } -layerChangeRange { M1 M9 }

puts "INFO: Power planning done."

# ----------------------------------------------------------------
# 4. Placement
#
# Density reduced 0.65 → 0.60 to leave routing headroom for the
# 144-bit weight broadcast net (1024 fan-out loads).  The placer
# needs slack to cluster the pe_array tiles and keep that net short.
# ----------------------------------------------------------------
setPlaceDensity 0.60
place_design
checkPlace $PNR/check_place.rpt
puts "INFO: Placement done."

optDesign -preCTS

# ----------------------------------------------------------------
# 5. Clock tree synthesis (CCOpt)
#
# target_skew relaxed 0.04 → 0.06 ns: the 32×32 clock tree drives
# ~1024× more flip-flops across a 2600 µm die.  40 ps is physically
# unreachable at this scale; 60 ps is aggressive but achievable.
# target_max_trans kept at 0.12 ns — cell slew limit unchanged.
# ----------------------------------------------------------------
create_ccopt_clock_tree_spec -file $BASE/pnr/ccopt.spec
set_ccopt_property target_max_trans 0.12
set_ccopt_property target_skew      0.06
ccopt_design
report_ccopt_clock_trees > $PNR/cts_report.rpt
puts "INFO: CTS done."

optDesign -postCTS -hold

# ----------------------------------------------------------------
# 6. Route
# ----------------------------------------------------------------
setNanoRouteMode -routeWithTimingDriven      true
setNanoRouteMode -routeWithSiDriven          true
setNanoRouteMode -routeInsertAntennaDiode    true
setNanoRouteMode -routeUseAutoVia            true
setNanoRouteMode -drouteUseMultiCutViaEffort medium

routeDesign

addFiller \
    -cell { SAEDRVT14_FILL64 SAEDRVT14_FILL32 SAEDRVT14_FILL16 \
            SAEDRVT14_FILL5  SAEDRVT14_FILL4  SAEDRVT14_FILL3 \
            SAEDRVT14_FILL2 } \
    -prefix FILL

puts "INFO: Routing done."

# Checkpoint right after clean routing.
saveDesign $PNR/top_ti_32x32_post_route.enc

optDesign -postRoute -setup -hold

# ----------------------------------------------------------------
# 7. Verify
# ----------------------------------------------------------------
verifyConnectivity -type all  -report $PNR/verify_connectivity.rpt
verifyGeometry                -report $PNR/verify_geometry.rpt
verifyProcessAntenna          -report $PNR/verify_antenna.rpt

# ----------------------------------------------------------------
# 8. Reports
# ----------------------------------------------------------------
report_timing -max_paths 20 -setup       > $PNR/timing_setup_final.rpt
report_timing -max_paths 10 -hold        > $PNR/timing_hold_final.rpt
report_area                              > $PNR/area_final.rpt
report_power -analysis_view typical_view > $PNR/power_final.rpt

puts "\n========== TIMING SUMMARY =========="
report_timing -max_paths 1 -setup
puts "====================================="

# ----------------------------------------------------------------
# 9. Outputs
# ----------------------------------------------------------------
write_netlist $PNR/top_ti_32x32_pnr.v
write_sdf -timescale ns -edges check_edge $PNR/top_ti_32x32_pnr.sdf
defOut $PNR/top_ti_32x32_pnr.def

streamOut $PNR/top_ti_32x32.gds \
    -mapFile $PDK/tech/milkyway/saed14nm_1p9m_gdsout.map \
    -libName top_ti_32x32 \
    -units   1000 \
    -mode    ALL

saveDesign $PNR/top_ti_32x32_pnr.enc

puts "\n==== P&R COMPLETE. Results in: $PNR ===="
