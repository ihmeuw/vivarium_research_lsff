"""
Module to facilitate using GBD id's in the shared functions.
"""
# prepend imports with underscores so they don't show up if id_helper module is imported using import *
from db_queries import get_ids as _get_ids
from pandas import DataFrame as _DataFrame

# The following list of valid entities was retrieved on 2020-10-12 from the hosted documentation:
# https://scicomp-docs.ihme.washington.edu/db_queries/current/get_ids.html
_entities = [
 'age_group',
 'age_group_set',
 'cause',
 'cause_set',
 'cause_set_version',
 'covariate',
 'decomp_step',
 'gbd_round',
 'healthstate',
 'indicator_component',
 'life_table_parameter',
 'location',
 'location_set',
 'location_set_version',
 'measure',
 'metric',
 'modelable_entity',
 'sdg_indicator',
 'sequela',
 'sequela_set',
 'sequela_set_version',
 'sex',
 'split',
 'study_covariate',
 'rei',
 'rei_set',
 'rei_set_version',
 'year'
]

def get_entities(source=None):
    """Returns a list the entities that are valid arguments to `get_ids()`. Entities are represented as strings.
    
    If source is None (default):
    Returns the 28 entities listed as valid arguments to `get_ids()` in the online documentation on 2020-10-12.
    https://scicomp-docs.ihme.washington.edu/db_queries/current/get_ids.html
    
    If source is 'docstring':
    Returns the entities listed as valid arguments in the docstring of `get_ids()`.
    As of 2020-10-12, there were only 22 entities listed in the docstring, whereas 28 entities were listed
    in the online documentation; those are accessible via get_entities() with the default source=None.
    
    Currently no other entity sources are supported (a ValueError will be raised if another value is passed).
    """
    if source is None:
        entities = _entities
    elif source == 'docstring':
        docstring = _get_ids.__doc__
        # This simplistic solution works with the current version, but it may need to be updated
        # to a more robust solution if the docstring changes...
        entities = docstring[docstring.find('[')+1:docstring.find(']')].split()
    else:
        raise ValueError(f"Unknonwn source of valid entities for `get_ids`: {source}")
    return entities

def find_anomalous_name_columns(entities):
    """Lists columns of entity tables that do not conatin a column called f'{entity}_name'."""
    # Use temporary dict to avoid calling _get_ids() twice in dictionary comprehension (or use a walrus := instead!)
    entities_columns = {entity: _get_ids(entity).columns for entity in entities}
    return {entity: columns for entity, columns in entities_columns.items() if f'{entity}_name' not in columns}
    # Better solution using walrus operator, but requires Python version 3.8 (https://www.python.org/dev/peps/pep-0572/):
#     return {entity: columns for entity in entities if f'{entity}_name' not in (columns:=_get_ids(entity).columns)}

def get_name_column(entity):
    """Returns the name column for the entity in the entity id table."""
    if entity=='year':
        return 'year_id' # Year table has only one column, 'year_id'
    elif entity=='life_table_parameter':
        return 'parameter_name'
    elif entity in [
        'cause_set_version', 'gbd_round', 'location_set_version',
        'sequela_set_version', 'sex', 'study_covariate', 'rei_set_version'
    ]: 
        return entity
    else:
        return f'{entity}_name'

# Should this have a parameter to optionally igonre NaN's? See comment for ids_to_names below.
# Other possible parameters:
#   allow duplicate names (True - list all id's that match the name, False - raise an exception)
#   deal with missing names (omit names that aren't in database, raise KeyError, fill with NaN)
# Also, it may be useful to enable passing a DataFrame instead of an entity,
# so that it's possible to filter the database first to avoid multiple matching names.
# Ok, I think I can get all the desired options by right-merging the names onto the id dataframe,
# then using logic on the options to remove things or raise exceptions as necessary.
def names_to_ids(entity, *entity_names):
    """Returns a pandas Series mapping entity names to entity id's for the specified GBD entity."""
    ids = _get_ids(entity)
    entity_name_col = get_name_column(entity)
    if len(entity_names)>0:
        ids = ids.query(f'{entity_name_col} in {entity_names}')
    # Year table only has one column, so we copy it
    if entity=='year':
        entity_name_col = 'year'
        ids[entity_name_col] = ids['year_id']
    return ids.set_index(entity_name_col)[f'{entity}_id']

# Should this have a parameter to optionally igonre NaN's?
# I got an error when I tried to pass entity_ids directly from a DataFrame that contained NaN's.
# Other possible parameters:
#   allow duplicate id's (True - list matching id's times, False - raise an exception)
#   deal with missing ids (omit id's that aren't in database, raise KeyError, fill with NaN)
def ids_to_names(entity, *entity_ids):
    """Returns a pandas Series mapping entity id's to entity names for the specified GBD entity."""
    ids = _get_ids(entity)
    if len(entity_ids)>0:
        # I think this raises an exception (KeyError and/or UndefinedVariableError) if entity_ids contains NaN
        ids = ids.query(f'{entity}_id in {entity_ids}')
    entity_name_col = get_name_column(entity)
    # Year table only has one column, so we copy it
    if entity=='year':
        entity_name_col = 'year'
        ids[entity_name_col] = ids['year_id']
    return ids.set_index(f'{entity}_id')[entity_name_col]

def process_singleton_ids(ids, entity):
    """Returns a single id if len(ids)==1. If len(ids)>1, returns ids (assumed to be a list), or raises
    a ValueError if the shared functions expect a single id rather than a list for the specified entity.
    """
    if len(ids)==1:
        ids = ids[0]
    elif entity=='gbd_round': # Also version id's?
        # It might be better to just let the shared functions raise an exception
        # rather than me doing it for them. In which case this function would be almost pointless...
        raise ValueError(f"Only single {entity} id's are allowed in shared functions.")
    return ids

def list_ids(entity, *entity_names):
    """Returns a list of ids (or a single id) for the specified entity names,
    suitable for passing to GBD shared functions.
    """
    # Series.to_list() converts to a list of Python int rather than numpy.int64
    # Conversion to the list type and the int type are both necessary for the shared functions
    ids = names_to_ids(entity, *entity_names).to_list()
    ids = process_singleton_ids(ids, entity)
    return ids

def get_entity_and_id_colname(table):
    """Returns the entity and entity id column name from an id table,
    assuming the entity id column name is f'{entity}_id',
    and that this is the first (or only) column ending in '_id'.
    """
#     id_colname = table.columns[table.columns.str.contains(r'\w+_id$')][0]
    id_colname = table.filter(regex=r'\w+_id$').columns[0]
    entity = id_colname[:-3]
    return entity, id_colname

def get_entity(table):
    """Returns the entity represented by a given id table,
    assuming the id column name is f'{entity}_id',
    and that this is the first (or only) column ending in '_id'.
    """
    return get_entity_and_id_colname(table)[0]

def get_id_colname(table):
    """Returns the entity id column name in the given id table,
    assuming it is the first (or only) column name that ends with '_id'.
    """
    return get_entity_and_id_colname(table)[1]

def ids_in(table):
    """Returns the ids in the given dataframe, either as a list of ints or a single int."""
    entity, id_colname = get_entity_and_id_colname(table)
    # Series.to_list() converts to a list of Python int rather than numpy.int64
    # Conversion to the list type and the int type are both necessary for the shared functions
    ids = table[id_colname].to_list()
    ids = process_singleton_ids(ids, entity)
    return ids

def search_id_table(table_or_entity, pattern, search_col=None, return_all_columns=False, **kwargs_for_contains):
    """Searches an entity id table for entity names matching the specified pattern, using pandas.Series.str.contains()."""
    if isinstance(table_or_entity, _DataFrame):
        df = table_or_entity
        entity = get_entity(df)
    elif isinstance(table_or_entity, str):
        entity = table_or_entity
        df = _get_ids(entity, return_all_columns)
    else:
        raise TypeError(f'Expecting type {_DataFrame} or {str} for `table_or_entity`. Got type {type(table_or_entity)}.')

    if search_col is None:
        search_col = get_name_column(entity)

    return df[df[search_col].str.contains(pattern, **kwargs_for_contains)]

def find_ids(table_or_entity, pattern, search_col=None, return_all_columns=False, **kwargs_for_contains):
    """Searches an entity id table for entity names matching the specified pattern, using pandas.Series.str.contains(),
    and returns a list of ids (or a single id) for the specified entity names, suitable for passing to GBD shared functions.
    """
    df = search_id_table(table_or_entity, pattern, search_col=None, return_all_columns=False, **kwargs_for_contains)
    return ids_in(df)

def add_entity_names(df, *entities):
    """Adds a name column for each specified entity in the dataframe df, by merging on the entity_id column.
    If no entities are passed, a name column is added for each enttity id column found in the dataframe.
    Intended to be called on dataframes returned by the shared functions.
    Returns a new object (does not modify df in place).
    """
    if len(entities) == 0:
        entities = df.filter(regex=r'\w+_id$').columns.str.replace('_id', '')
    for entity in entities:
        if entity == 'year':  # Avoid error from trying to merge duplicate year_id columns
            df = df.assign(year=df['year_id'])
        else:
            df = df.merge(_get_ids(entity)[[f'{entity}_id', get_name_column(entity)]])
    return df

def drop_id_columns(df, *entities, keep=False, errors='raise'):
    """If `keep` is False (default), drops the id column for each specified entity in the dataframe df.
    Drops all id columns if no entities are passed (you should probably only do this if you have added
    the corresponding entity name for the relevant id columns).
    If `keep` is set to True, the passed entities are those ids to keep, and all others will be dropped.
    Intended to be called on dataframes returned by the shared functions.
    Returns a new object (does not modify df in place).
    """
    id_colnames = [f'{entity}_id' for entity in entities]
    if len(entities) == 0 or keep:
        all_id_colnames = df.filter(regex=r'\w+_id$').columns # If no entities passed, drop all id columns
        if len(entities) > 0: # entities are those to keep, not drop
            id_colnames = all_id_colnames.difference(id_colnames)
    return df.drop(columns=id_colnames, errors=errors)

def replace_ids_with_names(df, *entities, invert=False, errors='raise'):
    """Replaces entity id columns in df with corresponding entity name columns."""
    if invert:
        entities = df.filter(regex=r'\w+_id$').columns.difference([f'{entity}_id' for entity in entities])
    df = add_entity_names(df, *entities)
    df = drop_id_columns(df, *entities, errors=errors)
    return df

