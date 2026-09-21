#!/usr/bin/env python3
# Author: Ievgen Vovk (vovk@icrr.u-tokyo.ac.jp)

import argparse
import datetime
import pymongo
import json
import astropy.time
import matplotlib.dates as mdates

from matplotlib import pyplot


def get_entries(mongo_client, property_name, astropy_time=True, tstart=None, tstop=None):
    """
    Retrieves the data base entries for the specified property, 
    optionally converting time to astropy.time.Time instances.
    
    Parameters
    ----------
    mongo_client: pymongo.mongo_client.MongoClient
        MongoDB client to use
    property_name: str
        Name of the data base property to retrieve
    astropy_time: bool, optional
        Whether to convert the time stamps to astropy.time.Time .
        If False, the original (datetime.datetime) instances are kept.
        Defaults to True.
    tstart: datetime.datetime, optional
        Start date for the return entries. If None, tstart condition will be ignored.
        Defaults to None.
    tstop: datetime.datetime, optional
        Stop date for the return entries. If None, tstop condition will be ignored.
        Defaults to None.
        
    Returns
    -------
    entries: dict
        Dictionary containing:
            "name": the name of the property (property_name argument)
            "time": list of time stamps (datetime.datetime or astropy.time.Time)
            "values": list of corresponding values
    """
    
    property_collection = mongo_client["bridgesmonitoring"]["properties"]
    chunk_collection = mongo_client["bridgesmonitoring"]["chunks"]
    
    descriptors = property_collection.find({"property_name": property_name})
    
    entries = {
        "name": property_name,
        "time": [],
        "value": []
    }

    for desc in descriptors:
        query = {"pid": desc["_id"]}
        
        if tstart is not None:
            query.update({"begin": {"$gte": tstart}})
        
        if tstop is not None:
            query.update({"end": {"$lte": tstop}})
            
        chunks = chunk_collection.find(query)
        
        for chunk in chunks:
            for value in chunk["values"]:
                entries["time"].append(value["t"])
                entries["value"].append(value["val"])
            
    if astropy_time:
        isodate = [t.isoformat() for t in entries["time"]]
        entries["time"] = astropy.time.Time(isodate, format="isot", scale="utc")
    
    return entries


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(description="""
    LST1 database query tool.
    Database entries are queried by their name with an optional time constraint.
    Start / stop times should be given in the ISO format, e.g. "2020-11-06" or "2020-11-06T18:00:00".
    """)

    arg_parser.add_argument("--name", default="",
                            help="property name to query")
    
    arg_parser.add_argument("--list-available", action="store_true",
                            help="list all available properties")
    
    arg_parser.add_argument("--tstart", default="2020-01-01",
                            help="start date to display")
    
    arg_parser.add_argument("--tstop", default="2030-01-01",
                            help="stop date to display")
    
    arg_parser.add_argument("--nofig", action="store_true",
                            help="disable the image generation")
    
    arg_parser.add_argument("--saveas", default="",
                            help="output image name. If not given, a window with the image will be shown at the screen.")
    
    arg_parser.add_argument("--dumpto", default="",
                            help="output JSON file name. If not given, the retrieved data points will not be stored to disk.")

    args = arg_parser.parse_args()
    
    client = pymongo.MongoClient("lst101-int:27017") #("localhost:27017")
    
    if args.list_available:
        #print(f"--- {'Summary':^40s} ---")
        #print(f"--- {'(entries in collections per data base)':^40s} ---")
        
        ## Querying the data base structure to get the summary
        #for db in client.list_database_names():
            #collections = client[db].list_collection_names()
            #document_count = [
                #client[db][coll].count_documents({}) 
                #for coll in collections
            #]
            
            #collections_with_document_count = list(zip(collections, document_count))
            #print(f"Database '{db:.<20s}': {collections_with_document_count}")
        
        # Listing available properties
        property_collection = client["bridgesmonitoring"]["properties"]
        entries = property_collection.find()
        
        property_names = list({entry["property_name"] for entry in entries})
        property_names.sort()
        
        print(f"--- {'':^40s} ---")
        print(f"--- {'Available property names':^40s} ---")
        print(f"--- {'':^40s} ---")
        
        for i, name in enumerate(property_names):
            print(f"{i:3} : {name}")
    
    # Retrieving a specific property
    if args.name != "":
        query = get_entries(
            client, 
            args.name, 
            tstart=datetime.datetime.fromisoformat(args.tstart),
            tstop=datetime.datetime.fromisoformat(args.tstop)
        )
        
        # Saving if requested
        if args.dumpto != "":
            data = {
                "mjd": query["time"].mjd.tolist(),
                "value": query["value"],
            }
            json.dump(data, open(args.dumpto, "w"))
    
        # Plotting if needed
        if not args.nofig:
            pyplot.xlabel("UTC")
            pyplot.ylabel("Value")
            pyplot.scatter(
                query["time"].datetime, 
                query["value"], 
                label=query["name"]
            )
            xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
            pyplot.gca().xaxis.set_major_formatter(xfmt)
            #pyplot.gcf().autofmt_xdate()
            
            tmin = min(
                datetime.datetime.fromisoformat(args.tstart),
                query["time"].datetime.min()
            )
            
            tmax = min(
                datetime.datetime.fromisoformat(args.tstop),
                query["time"].datetime.max()
            )
            
            pyplot.xlim(tmin, tmax)
            
            pyplot.legend()
            pyplot.grid(linestyle=":")
            
            if args.saveas != "":
                pyplot.savefig(args.saveas)
            else:
                pyplot.show()
    
