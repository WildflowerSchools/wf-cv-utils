import minimal_honeycomb
import pandas as pd
import re

def fetch_assignment_id_lookup(
    assignment_ids,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None

):
    assignment_ids = list(assignment_ids)
    if client is None:
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    result = client.bulk_query(
        request_name='searchAssignments',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'assignment_id',
                    'operator': 'IN',
                    'values': assignment_ids
                }
            }
        },
        return_data = [
            'assignment_id',
            {'assigned': [
                {'... on Device': [
                    'device_id'
                ]},
                {'... on Person': [
                    'person_id'
                ]},
                {'... on Material': [
                    'material_id'
                ]},
                {'... on Tray': [
                    'tray_id'
                ]},
            ]}
        ],
        id_field_name='assignment_id'
    )
    if len(result) == 0:
        return None
    records = list()
    for datum in result:
        records.append({
        'assignment_id': datum.get('assignment_id'),
        'device_id': datum.get('assigned').get('device_id'),
        'person_id': datum.get('assigned').get('person_id'),
        'material_id': datum.get('assigned').get('material_id'),
        'tray_id': datum.get('assigned').get('tray_id')
        })
    df = pd.DataFrame.from_records(records)
    df.set_index('assignment_id', inplace=True)
    return df


def extract_honeycomb_id(string):
    id = None
    m = re.search(
        '(?P<id>[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12})',
        string
    )
    if m:
        id = m.group('id')
    return id
