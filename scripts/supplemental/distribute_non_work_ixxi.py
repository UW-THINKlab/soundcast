﻿import array as _array
import os, sys
import pandas as pd
import h5py
import numpy as np
import inro.emme.matrix as ematrix
sys.path.append(os.path.join(os.getcwd(), "scripts"))
sys.path.append(os.path.join(os.getcwd(), "scripts/trucks"))
sys.path.append(os.getcwd())

def load_skims(skim_file_loc, mode_name, divide_by_100=False):
    """Loads H5 skim matrix for specified mode."""
    with h5py.File(skim_file_loc, "r") as f:
        skim_file = f["Skims"][mode_name][:]
    # Divide by 100 since decimals were removed in H5 source file through multiplication
    if divide_by_100:
        return skim_file.astype(float) / 100
    else:
        return skim_file


def calc_fric_fac(cost_skim, dist_skim, _coeff_df, zone_lookup, state):
    """Calculate friction factors for all trip purposes"""
    friction_fac_dic = {}
    for index, row in _coeff_df.iterrows():
        MIN_EXTERNAL_INDEX = zone_lookup[state.emme_settings.MIN_EXTERNAL]
        friction_fac_dic[row["purpose"]] = np.exp(
            (row["coefficient_value"])
            * (
                cost_skim
                + (dist_skim * state.emme_settings.autoop * state.emme_settings.avotda)
            )
        )
        ## Set external zones to zero to prevent external-external trips
        friction_fac_dic[row["purpose"]][MIN_EXTERNAL_INDEX:, MIN_EXTERNAL_INDEX:] = 0
        # friction_fac_dic[row['purpose']][:,[x for x in range(MIN_EXTERNAL_INDEX, len(cost_skim))]] = 0

    return friction_fac_dic


def delete_matrices(my_project, matrix_type):
    """Deletes all Emme matrices of specified type in emmebank"""
    for matrix in my_project.bank.matrices():
        if matrix.type == matrix_type:
            my_project.delete_matrix(matrix)


def load_matrices_to_emme(trip_table_in, trip_purps, fric_facs, my_project):
    """Loads data to Emme matrices: Ps and As and friction factor by trip purpose.
    Also initializes empty trip distribution and O-D result tables."""

    matrix_name_list = [matrix.name for matrix in my_project.bank.matrices()]
    zonesDim = len(my_project.current_scenario.zone_numbers)
    zones = my_project.current_scenario.zone_numbers

    # Create Emme matrices if they don't already exist
    for purpose in trip_purps:
        print(purpose)
        if purpose + "pro" not in matrix_name_list:
            my_project.create_matrix(
                str(purpose) + "pro", str(purpose) + " productions", "ORIGIN"
            )
        if purpose + "att" not in matrix_name_list:
            my_project.create_matrix(
                str(purpose) + "att", str(purpose) + " attractions", "DESTINATION"
            )
        if purpose + "fri" not in matrix_name_list:
            my_project.create_matrix(
                str(purpose) + "fri", str(purpose) + "friction factors", "FULL"
            )
        if purpose + "dis" not in matrix_name_list:
            my_project.create_matrix(
                str(purpose) + "dis", str(purpose) + "distributed trips", "FULL"
            )
        if purpose + "od" not in matrix_name_list:
            my_project.create_matrix(
                str(purpose) + "od", str(purpose) + "O-D tables", "FULL"
            )

        for p_a in ["pro", "att"]:
            # Load zonal production and attractions from CSV (output from trip generation)
            trips = np.array(trip_table_in[purpose + p_a].reindex(zones, fill_value=0))
            # code below does not work for GQs because there are only 3700 records in the csv file. Not sure if code above is ideal.
            # trips = np.array(trip_table_in.loc[0:zonesDim - 1][purpose + p_a])    # Less 1 because NumPy is 0-based\
            matrix_id = my_project.bank.matrix(purpose + p_a).id
            emme_matrix = my_project.bank.matrix(matrix_id)
            emme_matrix = ematrix.MatrixData(
                indices=[zones], type="f"
            )  # Access Matrix API

            # Update Emme matrix data
            emme_matrix.raw_data = _array.array(
                "f", trips
            )  # set raw matrix data equal to prod/attr data
            my_project.bank.matrix(matrix_id).set_data(
                emme_matrix, my_project.current_scenario
            )

        # Load friction factors by trip purpose
        fri_fac = fric_facs[purpose][0 : zonesDim + 1, 0 : zonesDim + 1]
        emme_matrix = ematrix.MatrixData(
            indices=[zones, zones], type="f"
        )  # Access Matrix API
        emme_matrix.raw_data = [_array.array("f", row) for row in fri_fac]
        matrix_id = my_project.bank.matrix(purpose + "fri").id
        my_project.bank.matrix(matrix_id).set_data(
            emme_matrix, my_project.current_scenario
        )


def balance_matrices(trip_purps, state):
    """Balances productions and attractions by purpose for all internal zones"""

    for purpose in trip_purps:
        # For friction factors, make sure 0s in Externals are actually 0 and not fractional to avoid intrazonal trips
        state.main_project.matrix_calculator(
            result="mf" + purpose + "fri",
            expression="0",
            constraint_by_zone_destinations=str(state.emme_settings.MIN_EXTERNAL)
            + "-"
            + str(state.emme_settings.MAX_EXTERNAL),
            constraint_by_zone_origins=str(state.emme_settings.MIN_EXTERNAL)
            + "-"
            + str(state.emme_settings.MAX_EXTERNAL),
        )
        print("Balancing non-work external trips, for purpose: " + str(purpose))
        state.main_project.matrix_balancing(
            results_od_balanced_values="mf" + purpose + "dis",
            od_values_to_balance="mf" + purpose + "fri",
            origin_totals="mo" + purpose + "pro",
            destination_totals="md" + purpose + "att",
            constraint_by_zone_destinations="1-"
            + str(state.emme_settings.MAX_EXTERNAL),
            constraint_by_zone_origins="1-" + str(state.emme_settings.MAX_EXTERNAL),
        )


def calculate_daily_trips_externals(trip_purps, my_project):
    """Transpose matrices to get return trips (internal-external -> external-internal)"""

    for purpose in trip_purps:
        my_project.matrix_calculator(
            result="mf" + purpose + "od",
            expression="mf" + purpose + "dis + mf" + purpose + "dis" + "'",
        )


def distribute_trips_externals(trip_table_in, trip_purps, fric_facs, state):
    """Load data in Emme, balance trips by purpose, and produce O-D trip tables"""

    # Clear all existing matrices
    delete_matrices(state.main_project, "ORIGIN")
    delete_matrices(state.main_project, "DESTINATION")
    delete_matrices(state.main_project, "FULL")

    # Load data into fresh Emme matrices
    load_matrices_to_emme(trip_table_in, trip_purps, fric_facs, state.main_project)

    # Balance matrices
    balance_matrices(trip_purps, state)

    # Calculate daily trips
    calculate_daily_trips_externals(trip_purps, state.main_project)


def emme_matrix_to_np(trip_purp_list, state):
    """Export results from emme to numpy, for external zones only.
    Returns dictionary of arrays with keys as trip purpose."""

    trips_by_purpose = {}
    for purpose in trip_purp_list:
        # Load Emme O-D total trip data by purpose
        matrix_id = state.main_project.bank.matrix(purpose + "od").id
        emme_matrix = state.main_project.bank.matrix(matrix_id)
        emme_data = emme_matrix.get_data()  # Access emme data as numpy matrix
        emme_data = np.array(emme_data.raw_data, dtype="float64")
        filtered = np.zeros_like(emme_data)

        # Add only external rows and columns from emme data
        filtered[state.emme_settings.HIGH_TAZ :, :] = emme_data[
            state.emme_settings.HIGH_TAZ :, :
        ]
        filtered[:, state.emme_settings.HIGH_TAZ :] = emme_data[
            :, state.emme_settings.HIGH_TAZ :
        ]
        trips_by_purpose[purpose] = filtered

    return trips_by_purpose


def main(state):
    # Load the trip productions and attractions
    trip_table = pd.read_csv(
        state.emme_settings.trip_table_loc, index_col="taz"
    )  # total 4K Ps and As by trip purpose

    # Import gravity model coefficients by trip purpose from db
    coeff_df = pd.read_sql("SELECT * FROM gravity_model_coefficients", con=state.conn)

    # All Non-work external trips assumed as single purpose HSP (home-based shopping trips)
    trip_purpose_list = ["hsp"]

    output_dir = os.path.join(os.getcwd(), r"outputs\supplemental")

    # my_project = state.main_project
    if state.main_project.data_explorer.active_database().title() != "Supplementals":
        state.main_project.change_active_database("Supplementals")

    # global dictZoneLookup
    dictZoneLookup = dict(
        (value, index)
        for index, value in enumerate(state.main_project.current_scenario.zone_numbers)
    )

    # Load skim data
    am_cost_skim = load_skims(
        f"inputs/model/{state.input_settings.abm_model}/roster/{state.emme_settings.am_skim_name}.h5",
        mode_name="sov_inc2g",
    )
    am_dist_skim = load_skims(
        f"inputs/model/{state.input_settings.abm_model}/roster/{state.emme_settings.am_skim_name}.h5",
        mode_name="sov_inc1d",
        divide_by_100=True,
    )
    pm_cost_skim = load_skims(
        f"inputs/model/{state.input_settings.abm_model}/roster/{state.emme_settings.pm_skim_name}.h5",
        mode_name="sov_inc2g",
    )
    pm_dist_skim = load_skims(
        f"inputs/model/{state.input_settings.abm_model}/roster/{state.emme_settings.pm_skim_name}.h5",
        mode_name="sov_inc1d",
        divide_by_100=True,
    )
    # Average skims between AM and PM periods
    cost_skim = (am_cost_skim + pm_cost_skim) * 0.5
    dist_skim = (am_dist_skim + pm_dist_skim) * 0.5

    # Compute friction factors by trip purpose
    fric_facs = calc_fric_fac(cost_skim, dist_skim, coeff_df, dictZoneLookup, state)

    # Create trip table for externals
    distribute_trips_externals(trip_table, trip_purpose_list, fric_facs, state)

    # Export results as array to write to h5 file
    ixxi_trips = emme_matrix_to_np(trip_purpose_list, state)

    # Distribute trips by auto mode, taken from observed splits
    # All ixxi trips are HSP purpose
    # Export to h5 container
    ixxi_trips = ixxi_trips["hsp"]

    ixxi_mode_share_df = pd.read_sql("SELECT * FROM ixxi_mode_share", con=state.conn)
    ixxi_h5 = h5py.File(os.path.join(output_dir, "external_non_work.h5"), "w")

    for mode in ["sov", "hov2", "hov3"]:
        mode_share = ixxi_mode_share_df.loc[
            ixxi_mode_share_df["mode"] == mode, "ixxi_mode_share"
        ].values[0]
        ixxi_data = ixxi_trips * mode_share
        ixxi_h5.create_dataset(mode, data=ixxi_data)

    ixxi_h5.close()

if __name__ == "__main__":
    main()
