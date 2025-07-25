import array as _array
import inro.emme.desktop.app as app
import inro.modeller as _m
import inro.emme.matrix as ematrix
import inro.emme.database.matrix
import inro.emme.database.emmebank as _eb
import json
import numpy as np
import pandas as pd
import time
import os, sys
import h5py
from sqlalchemy import create_engine

sys.path.append(os.path.join(os.getcwd(), "inputs"))
sys.path.append(os.path.join(os.getcwd(), "scripts"))
sys.path.append(os.getcwd())
# from truck_model import *
from scripts.emme_project import *

# from truck_configuration import *
# from emme_configuration import *
# from input_configuration import *
import toml

# from settings import run_args
# from scripts.settings import state
from pathlib import Path

# state = state.generate_state(run_args.args.configs_dir)

# config = toml.load(os.path.join(os.getcwd(), "configuration/input_configuration.toml"))
# network_config = toml.load(
#     os.path.join(os.getcwd(), "configuration/network_configuration.toml")
# )


def network_importer(state):
    for scenario in list(state.main_project.bank.scenarios()):
        state.main_project.bank.delete_scenario(scenario)

    # create scenario
    state.main_project.bank.create_scenario(1002)
    state.main_project.change_scenario()
    state.main_project.delete_links()
    state.main_project.delete_nodes()
    state.main_project.process_modes("inputs/scenario/networks/modes.txt")
    state.main_project.process_base_network(
        "inputs/scenario/networks/roadway/" + state.network_settings.truck_base_net_name
    )


def json_to_dictionary(dict_name, state):
    # Determine the Path to the input files and load them
    input_filename = os.path.join(
        f"inputs/model/{state.input_settings.abm_model}/trucks/", dict_name + ".txt"
    ).replace("\\", "/")
    my_dictionary = json.load(open(input_filename))
    return my_dictionary


def write_truck_trips(EmmeProject, state):
    truck_od_matrices = ["medtrk", "hvytrk", "deltrk"]

    # if h5 exists, delete it and re-write
    try:
        os.remove(state.network_settings.truck_trips_h5_filename)
    except OSError:
        pass

    my_store = h5py.File(state.network_settings.truck_trips_h5_filename, "w")
    for tod in state.network_settings.tods:
        my_store.create_group(tod)
        for name in truck_od_matrices:
            matrix_name = "mf" + tod + "_" + name + "_trips"
            matrix_id = EmmeProject.bank.matrix(matrix_name).id
            matrix = EmmeProject.bank.matrix(matrix_id)
            matrix_value = np.matrix(matrix.raw_data)
            my_store[tod].create_dataset(
                matrix_name, data=matrix_value.astype("float32"), compression="gzip"
            )
            matrix_value = None

    my_store.close()


def create_matrices(my_project, truck_matrix_df):
    for matrix_type in ["scalar", "origin", "destination", "full"]:
        df = truck_matrix_df[truck_matrix_df["matrix_type"] == matrix_type]
        for index, row in df.iterrows():
            my_project.create_matrix(
                row["matrix_name"], row["description"], matrix_type.upper()
            )


def load_data_to_emme(balanced_prod_att, my_project, zones, state):
    """Populate Emme matrices with medium and heavy truck productions and attractions."""

    for truck_type in ["m", "h", "d"]:  # Loop through medium (m) and heavy (h) trucks
        for datatype in ["pro", "att"]:
            col_values = np.zeros(len(zones))
            numpy_data = balanced_prod_att[truck_type + "tk" + datatype].values
            col_values[: len(numpy_data)] = numpy_data
            mat_name = "mo" + truck_type + "t" + datatype
            matrix_id = my_project.bank.matrix(str(mat_name)).id
            my_project.bank.matrix(matrix_id).set_numpy_data(
                col_values, my_project.current_scenario
            )

            # Transpose Attractions (Destination Matrices are populated)
            if datatype == "att":
                my_project.matrix_calculator(
                    result="md" + truck_type + "tatt",
                    expression="mo" + truck_type + "tatt" + "'",
                )

    # Add operating costs as scalar matrices
    # Calculate operating costs for model year
    op_cost_df = pd.read_sql(
        "SELECT * FROM truck_inputs WHERE data_type='operating_costs'", con=state.conn
    )

    data_year = int(op_cost_df["year"][0])
    if data_year < int(state.input_settings.model_year):
        growth_rate = 1 + (
            state.network_settings.truck_operating_cost_rate * (int(state.input_settings.model_year) - data_year)
        )
        op_cost_df["cents_per_mile"] = op_cost_df["value"] * growth_rate

    # Note: Using medium truck coefficients for delivery trucks
    for mat_name, truck_type in {
        "msmedop": "medium",
        "mshvyop": "heavy",
        "msdelop": "medium",
    }.items():
        op_cost = (
            op_cost_df[op_cost_df["truck_type"] == truck_type]["cents_per_mile"]
            .astype("float")
            .values[0]
        )
        print(op_cost)
        print(matrix_id)
        matrix_id = my_project.bank.matrix(str(mat_name)).id
        my_project.bank.matrix(matrix_id).set_numpy_data(
            op_cost, my_project.current_scenario
        )


def import_skims(my_project, input_skims, zones, zonesDim, state):
    # Open GC skims from H5 container, average am/pm, import to emme:
    np_gc_skims = {}
    np_bidir_gc_skims = {}
    for tod in state.network_settings.truck_generalized_cost_tod.keys():
        hdf_file = h5py.File(
            f"inputs/model/{state.input_settings.abm_model}/roster/" + tod + ".h5", "r"
        )
        for item in input_skims.values():
            # gc
            skim_name = item["gc_name"]
            h5_skim = hdf_file["Skims"][skim_name]
            np_skim = np.matrix(h5_skim)
            np_gc_skims[
                skim_name + "_" + state.network_settings.truck_generalized_cost_tod[tod]
            ] = np_skim

            # distance
            skim_name = item["dist_name"]
            h5_skim = hdf_file["Skims"][skim_name]
            np_skim = np.matrix(h5_skim)
            np_gc_skims[
                skim_name + "_" + state.network_settings.truck_generalized_cost_tod[tod]
            ] = np_skim

    # zones = my_project.current_scenario.zone_numbers
    # zonesDim = len(my_project.current_scenario.zone_numbers)

    for truck_type in input_skims.values():
        # gc:
        am_skim_name = truck_type["gc_name"] + "_am"
        pm_skim_name = truck_type["gc_name"] + "_pm"
        bidir_skim_name = truck_type["gc_bidir_name"]
        bi_dir_skim = np_gc_skims[am_skim_name] + np_gc_skims[pm_skim_name]
        bi_dir_skim = np.asarray(bi_dir_skim)
        # have sum, now get average
        bi_dir_skim *= 0.5
        bi_dir_skim = bi_dir_skim[0:zonesDim, 0:zonesDim]
        np_bidir_gc_skims[bidir_skim_name] = bi_dir_skim

        # distance
        am_skim_name = truck_type["dist_name"] + "_am"
        pm_skim_name = truck_type["dist_name"] + "_pm"
        bidir_skim_name = truck_type["dist_bidir_name"]
        # distance skims are multiplied by 100 when exported by SkimsAndPaths, so we devide by 100
        bi_dir_skim = (np_gc_skims[am_skim_name] + np_gc_skims[pm_skim_name]) / 100.0
        bi_dir_skim = np.asarray(bi_dir_skim)
        # have sum, now get average
        bi_dir_skim *= 0.5
        bi_dir_skim = bi_dir_skim[0:zonesDim, 0:zonesDim]
        np_bidir_gc_skims[bidir_skim_name] = bi_dir_skim

    # import bi-directional skims to emmebank
    for mat_name, matrix in np_bidir_gc_skims.items():
        matrix_id = my_project.bank.matrix(str(mat_name)).id
        emme_matrix = ematrix.MatrixData(indices=[zones, zones], type="f")
        emme_matrix.raw_data = [_array.array("f", row) for row in matrix]
        my_project.bank.matrix(matrix_id).set_data(
            emme_matrix, my_project.current_scenario
        )


def balance_attractions(my_project, state):
    # Balance Medium Truck Attractions to productions:
    my_project.matrix_calculator(
        result="msmtprof", expression="momtpro", aggregation_origins="+"
    )
    my_project.matrix_calculator(
        result="msmtattf", expression="mdmtatt", aggregation_destinations="+"
    )
    my_project.matrix_calculator(
        result="msmtatfe",
        expression="mdmtatt",
        constraint_by_zone_destinations=str(state.network_settings.LOW_STATION)
        + "-"
        + str(state.network_settings.HIGH_STATION),
        aggregation_destinations="+",
    )
    my_project.matrix_calculator(
        result="mdmtatt",
        expression="mdmtatt * ((msmtprof - msmtatfe)/(msmtattf-msmtatfe))",
    )

    # Balance Heavy Truck Attractions to productions:
    my_project.matrix_calculator(
        result="mshtprof", expression="mohtpro", aggregation_origins="+"
    )
    my_project.matrix_calculator(
        result="mshtattf", expression="mdhtatt", aggregation_destinations="+"
    )
    my_project.matrix_calculator(
        result="mshtatfe",
        expression="mdhtatt",
        constraint_by_zone_destinations=str(state.network_settings.LOW_STATION)
        + "-"
        + str(state.network_settings.HIGH_STATION),
        aggregation_destinations="+",
    )
    my_project.matrix_calculator(
        result="mdhtatt",
        expression="mdhtatt * ((mshtprof - mshtatfe)/(mshtattf-mshtatfe))",
    )

    # Balance Delivery Attractions to productions:
    my_project.matrix_calculator(
        result="msdtprof", expression="modtpro", aggregation_origins="+"
    )
    my_project.matrix_calculator(
        result="msdtattf", expression="mddtatt", aggregation_destinations="+"
    )
    my_project.matrix_calculator(
        result="msdtatfe",
        expression="mddtatt",
        constraint_by_zone_destinations=str(state.network_settings.LOW_STATION)
        + "-"
        + str(state.network_settings.HIGH_STATION),
        aggregation_destinations="+",
    )
    my_project.matrix_calculator(
        result="mddtatt",
        expression="mddtatt * ((msdtprof - msdtatfe)/(msdtattf-msdtatfe))",
    )


def float_to_string(val):
    """Return string with fixed precision, removes scientific notation for small floats."""

    return "{:.6f}".format(val)


def calculate_impedance(my_project, state):
    coeff_df = pd.read_sql(
        "SELECT * FROM truck_inputs WHERE data_type='distribution_coeff'", con=state.conn
    )
    med_coeff = float_to_string(
        coeff_df[coeff_df["truck_type"] == "medium"]["value"].values[0]
    )
    hvy_coeff = float_to_string(
        coeff_df[coeff_df["truck_type"] == "heavy"]["value"].values[0]
    )
    # Using medium truck coefficients for delivery trucks
    del_coeff = float_to_string(
        coeff_df[coeff_df["truck_type"] == "medium"]["value"].values[0]
    )

    vot_df = pd.read_sql("SELECT * FROM truck_inputs WHERE data_type='vot'", con=state.conn)
    med_vot = float_to_string(
        vot_df[vot_df["truck_type"] == "medium"]["value"].values[0]
    )
    hvy_vot = float_to_string(
        vot_df[vot_df["truck_type"] == "heavy"]["value"].values[0]
    )
    # Using medium truck coefficients for delivery trucks
    del_vot = float_to_string(
        vot_df[vot_df["truck_type"] == "medium"]["value"].values[0]
    )

    # Load friction factor and value of time coefficients

    # set flag to 0 for external-external OD paris and all others equal to 1
    my_project.matrix_calculator(result="mfintflg", expression="1")
    # my_project.matrix_calculator(
    #     result="mfintflg",
    #     expression="0",
    #     constraint_by_zone_destinations=network_config["EXTERNAL_DISTRICT"],
    #     constraint_by_zone_origins=network_config["EXTERNAL_DISTRICT"],
    # )

    # calculate medium truck impedances:
    my_project.matrix_calculator(
        result="mfmedimp",
        expression="exp("
        + med_coeff
        + "*(mfbmedcs+(mfbmedds*msmedop*"
        + med_vot
        + ")))*mfintflg",
        constraint_by_zone_destinations="1-" + str(state.network_settings.HIGH_STATION),
        constraint_by_zone_origins="1-" + str(state.network_settings.HIGH_STATION),
    )

    # calculate heavy truck impedances:
    my_project.matrix_calculator(
        result="mfhvyimp",
        expression="exp("
        + hvy_coeff
        + "*(mfbhvycs+(mfbhvyds*mshvyop*"
        + hvy_vot
        + ")))*mfintflg",
        constraint_by_zone_destinations="1-" + str(state.network_settings.HIGH_STATION),
        constraint_by_zone_origins="1-" + str(state.network_settings.HIGH_STATION),
    )

    # calculate delivery truck impedances:
    my_project.matrix_calculator(
        result="mfdelimp",
        expression="exp("
        + hvy_coeff
        + "*(mfbdelcs+(mfbdelds*msdelop*"
        + del_vot
        + ")))*mfintflg",
        constraint_by_zone_destinations="1-" + str(state.network_settings.HIGH_STATION),
        constraint_by_zone_origins="1-" + str(state.network_settings.HIGH_STATION),
    )


def balance_matrices(my_project, state):
    # Balance Medium Trucks
    state.main_project.matrix_balancing(
        results_od_balanced_values="mfmeddis",
        od_values_to_balance="mfmedimp",
        origin_totals="momtpro",
        destination_totals="mdmtatt",
        constraint_by_zone_destinations="1-" + str(state.network_settings.HIGH_STATION),
        constraint_by_zone_origins="1-" + str(state.network_settings.HIGH_STATION),
    )
    # Balance Heavy Trucks
    state.main_project.matrix_balancing(
        results_od_balanced_values="mfhvydis",
        od_values_to_balance="mfhvyimp",
        origin_totals="mohtpro",
        destination_totals="mdhtatt",
        constraint_by_zone_destinations="1-" + str(state.network_settings.HIGH_STATION),
        constraint_by_zone_origins="1-" + str(state.network_settings.HIGH_STATION),
    )

    # Balance Delivery Trucks
    state.main_project.matrix_balancing(
        results_od_balanced_values="mfdeldis",
        od_values_to_balance="mfdelimp",
        origin_totals="modtpro",
        destination_totals="mddtatt",
        constraint_by_zone_destinations="1-" + str(state.network_settings.HIGH_STATION),
        constraint_by_zone_origins="1-" + str(state.network_settings.HIGH_STATION),
    )


def calculate_daily_trips(my_project, state):
    # Calculate Daily OD trips:
    # The distribution matrices (e.g. 'mfmeddis') are in PA format. Need to convert to OD format by transposing
    my_project.matrix_calculator(
        result="mfmedod", expression="0.5*mfmeddis + 0.5*mfmeddis" + "'"
    )
    my_project.matrix_calculator(
        result="mfhvyod", expression="0.5*mfhvydis + 0.5*mfhvydis" + "'"
    )
    my_project.matrix_calculator(
        result="mfdelod", expression="0.5*mfdeldis + 0.5*mfdeldis" + "'"
    )

    # convert annual external medium truck trips to daily and add to medium od
    my_project.matrix_calculator(
        result="mfmedod", expression="mfmedod + (mfmedee + mfmedei + mfmedie)/264"
    )

    # convert annual external heavy truck trips to daily and add to heavy od:
    my_project.matrix_calculator(
        result="mfhvyod", expression="mfhvyod + (mfhvyee + mfhvyei + mfhvyie)/264"
    )

    # convert annual external delivery truck trips to daily and add to heavy od:
    my_project.matrix_calculator(
        result="mfdelod", expression="mfdelod + (mfdelee + mfdelei + mfdelie)/264"
    )

    # apply vehicle-equivalency factors to medium and heavy trucks:
    my_project.matrix_calculator(result="mfmedod", expression="mfmedod * 1.5")
    my_project.matrix_calculator(result="mfhvyod", expression="mfhvyod * 2")
    my_project.matrix_calculator(result="mfdelod", expression="mfdelod * 1.5")

    # apply time of day factors:
    df_tod_factors = pd.read_sql(f"SELECT * FROM truck_time_of_day_factors WHERE model=='{state.input_settings.abm_model}'", con=state.conn)

    for tod in df_tod_factors["time_period"].unique():
        for truck_type, matrix_name in {
            "medtrk": "medod",
            "hvytrk": "hvyod",
            "deltrk": "delod",
        }.items():
            df = df_tod_factors[
                (df_tod_factors["time_period"] == tod)
                & (df_tod_factors["truck_type"] == truck_type)
            ]
            my_project.matrix_calculator(
                result="mf" + tod + "_" + truck_type + "_trips",
                expression="mf" + matrix_name + "*" + str(df["value"].values[0]),
            )


def write_summary(my_project):
    # Write production and attraction totals
    truck_pa = {"prod": {}, "attr": {}}

    for truck_type in ["mt", "ht", "dt"]:
        truck_pa["prod"][truck_type] = (
            my_project.bank.matrix("mo" + truck_type + "pro").get_numpy_data().sum()
        )
        truck_pa["attr"][truck_type] = (
            my_project.bank.matrix("md" + truck_type + "att").get_numpy_data().sum()
        )

    pd.DataFrame.from_dict(truck_pa).to_csv(r"outputs/trucks/trucks_summary.csv")


def main(state):
    # my_project = EmmeProject(network_config["truck_model_project"], state)
    # zones = my_project.current_scenario.zone_numbers
    if state.main_project.data_explorer.active_database().title() != "Supplementals":
        state.main_project.change_active_database("Supplmentals")
    state.main_project.change_active_database("TruckModel")

    input_skims = json_to_dictionary("input_skims", state)
    truck_matrix_list = pd.read_csv(
        f"inputs/model/{state.input_settings.abm_model}/trucks/truck_matrices.csv"
    )

    balanced_prod_att = pd.read_csv("outputs/supplemental/7_balance_trip_ends.csv")

    network_importer(state)
    zones = state.main_project.current_scenario.zone_numbers
    zonesDim = len(zones)

    # Load zone partitions (used to identify external zones)
    state.main_project.initialize_zone_partition("ga")
    state.main_project.process_zone_partition(
        f"inputs/model/{state.input_settings.abm_model}/trucks/"
        + state.network_settings.districts_file
    )

    state.main_project.delete_matrices("ALL")
    create_matrices(state.main_project, truck_matrix_list)
    load_data_to_emme(balanced_prod_att, state.main_project, zones, state)
    import_skims(state.main_project, input_skims, zones, zonesDim, state)
    balance_attractions(state.main_project, state)
    calculate_impedance(state.main_project, state)
    balance_matrices(state.main_project, state)
    calculate_daily_trips(state.main_project, state)
    write_truck_trips(state.main_project, state)
    write_summary(state.main_project)

if __name__ == "__main__":
    main()
