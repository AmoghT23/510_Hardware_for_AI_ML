# ============================================================
# run_genus.tcl  Cadence Genus 17.1  (legacy_ui)
# Design  : top_ti  (32×32 systolic array)
# PDK     : SAED14nm RVT
# Target  : 500 MHz, 2.0 ns
#
# Changes vs. single-tile run:
#   + ifm_buffer.sv and pe_array.sv added to RTL sources
#   + top_ti.sv renamed to top.sv (module name still top_ti)
#   + syn_generic/map effort raised to high (weight fanout opt)
#   + write_sdc added for Innovus import
#   + output files renamed to top_ti_32x32_*
# ============================================================

set BASE    /u/amoghvt/Desktop/HW-AI
set LIB_DIR /pkgs/synopsys/libs/saed14nm/stdcell_rvt/db_nldm
set RTL_DIR $BASE/rtl

# ── 1. Libraries ─────────────────────────────────────────────
set_attribute lib_search_path $LIB_DIR /
set_attribute library { saed14rvt_tt0p8v25c.lib \
                        saed14rvt_ss0p72v25c.lib \
                        saed14rvt_ff0p88v25c.lib } /

# ── 2. Read RTL ──────────────────────────────────────────────
# Dependency order: leaves first, top last.
# ifm_buffer and pe_array are new for the 32×32 design.
read_hdl -sv $RTL_DIR/sram_sp.sv
read_hdl -sv $RTL_DIR/grad_core.sv
read_hdl -sv $RTL_DIR/ifm_buffer.sv
read_hdl -sv $RTL_DIR/compute_core_ti.sv
read_hdl -sv $RTL_DIR/pe_array.sv
read_hdl -sv $RTL_DIR/interface_ti.sv
read_hdl -sv $RTL_DIR/top.sv

# ── 3. Elaborate ─────────────────────────────────────────────
elaborate top_ti
check_design

# ── 4. Constraints ───────────────────────────────────────────
# Same SDC as single-tile (500 MHz clock, same I/O delays).
read_sdc $BASE/constraints.sdc

# ── 5. Synthesize ────────────────────────────────────────────
# Raised generic and map effort to high vs. single-tile run.
# The 144-bit weight broadcast net (1024 loads) needs full
# Genus buffer-tree insertion — medium effort leaves fan-out
# violations that only high effort will fix via net splitting.
set_attribute syn_generic_effort high /
set_attribute syn_map_effort     high /
set_attribute syn_opt_effort     high /

syn_generic
syn_map
syn_opt

# ── 6. Reports ───────────────────────────────────────────────
file mkdir $BASE/reports
report_timing  -num_paths 10  > $BASE/reports/timing_32x32.rpt
report_area                   > $BASE/reports/area_32x32.rpt
report_power                  > $BASE/reports/power_32x32.rpt
report_gates                  > $BASE/reports/gates_32x32.rpt
catch { report_qor            > $BASE/reports/qor_32x32.rpt }

# Print the worst slack inline so you see it immediately in the log.
puts "\n==== WORST SETUP SLACK ===="
report_timing -num_paths 1
puts "==========================="

# ── 7. Write outputs ─────────────────────────────────────────
# write_hdl -pg  : includes power/ground pins — required by Innovus.
# write_sdc      : exports the elaborated+constrained SDC for Innovus
#                  (replaces the hand-written constraints.sdc at PnR stage).
write_hdl -pg                 > $BASE/reports/top_ti_32x32_netlist.v
write_sdc                     > $BASE/reports/top_ti_32x32.sdc
write_sdf -timescale ns       > $BASE/reports/top_ti_32x32.sdf

puts "\n==== Synthesis done. Check $BASE/reports/ ===="
puts "==== Key files for Innovus: ===="
puts "====   Netlist : $BASE/reports/top_ti_32x32_netlist.v ===="
puts "====   SDC     : $BASE/reports/top_ti_32x32.sdc      ===="
