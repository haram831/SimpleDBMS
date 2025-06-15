from lark import Lark, Transformer, Token, Tree
import sys
import os
import json
import re
from berkeleydb import db
from datetime import datetime


# DB 저장 디렉토리 및 환경 설정
DB_DIR = "./db"
os.makedirs(DB_DIR, exist_ok=True)

# 환경 초기화
env = db.DBEnv()
env.open(DB_DIR, db.DB_CREATE | db.DB_INIT_MPOOL)

# 메타데이터 DB 핸들러
metadata_db = db.DB(env)
metadata_db.open("metadata.db", None, db.DB_HASH, db.DB_CREATE)

# grammar.lark 파일 읽기
with open("grammar.lark", encoding="utf-8") as file:
    sql_parser = Lark(file.read(), start="command", lexer="basic")

prompt = "DB_2023-13751> "

def print_metadata_db():
    print("======== METADATA_DB CONTENTS ========")
    for key, value in metadata_db.items():
        table_name = key.decode("utf-8")
        schema = json.loads(value.decode("utf-8"))
        print(f"\n📄 Table: {table_name}")
        for col in schema:
            print(f"  - {col}")
    print("=======================================")

def parse_char_length(data_type_node):
    """Parse char length from data type node, properly extracting the value."""
    if data_type_node.children[0].type == 'TYPE_CHAR':
        # If it's a CHAR type, extract the length from INT token
        try:
            length = int(data_type_node.children[2].value)
            if length <= 0:
                return None
            return length
        except (IndexError, ValueError):
            return None
    return None

def get_data_type_str(data_type_node):
    """Generate the data type string based on the node."""
    if data_type_node.children[0].type == 'TYPE_INT':
        return "int"
    elif data_type_node.children[0].type == 'TYPE_DATE':
        return "date"
    elif data_type_node.children[0].type == 'TYPE_CHAR':
        length = parse_char_length(data_type_node)
        if length is not None:
            return f"char({length})"
    return None


def extract_from_tables(table_ref_list):
    tables = []
    join_clauses_list = []

    if hasattr(table_ref_list.children[0], 'data') and table_ref_list.children[0].data == 'referred_table':
        referred_table_node = table_ref_list.children[0]
        table_node = referred_table_node.children[0]
        table_name = table_node.children[0].value.lower()
        tables.append(table_name)

        #join 처리
        if len(referred_table_node.children) > 1:
            for join_clause in referred_table_node.children[1:]:
                if hasattr(join_clause, 'data') and join_clause.data == 'join_clause':
                    join_table_node = join_clause.children[1]
                    join_table_name = join_table_node.children[0].value.lower()
                    tables.append(join_table_name)
                    join_clauses_list.append(join_clause)

    return tables, join_clauses_list

def check_table_exist(table_list):
    for table in table_list:
        if not metadata_db.exists(table.encode("utf-8")):
            print(f"{prompt}Select has failed: {table} does not exist")
            return False

#where operation
def parse_table_col(tree, columns):
    table_name = None
    column_name = None
    col_type = None
    # column reference
    if tree.data == "column_reference": 
        column_name = tree.children[1].children[0].value
        # [table_name "."] 없는 경우
        if tree.children[0] is None:
            table_name = None
        else:
            table_name = tree.children[0].children[0].value
        
        for col in columns:
            if col['name'] == column_name and (not table_name or col['table'] == table_name):
                col_type = col['type']
        return table_name, column_name, col_type
    else:
        raise Exception("Invalid column refernece structure")
    
def parse_comp_val(tree):
    if tree.data == "comparable_value" and isinstance(tree.children[0], Token):
        return tree.children[0].value.strip('\'"')
    
def cast_value_to_column_type(type, value):
    try:
        if type == 'int':
            return int(value)
        elif type.startswith('char('):
            match = re.match(r'char\((\d+)\)', type)
            if match:
                max_length = int(match.group(1))
                return value[:max_length]

            return value
        elif type == 'date':
            return value
        else:
            print(f"{prompt}Unsupported column type: {type}")
            return None
    except Exception:
        print(f"{prompt}Type conversion error: cannot convert '{value}' to {type}")
        return None
    
def predicate_check(boolean_term: Tree, table_schemas: dict, columns, row):
    def check_expr(expr):
        def resolve_col_value(tree):
            if tree.data == "comp_operand":
                return resolve_col_value(tree.children[0]) 
            if tree.data == "column_reference":
                table_name, col_name, type = parse_table_col(tree, columns)
                matches = []
                #col_name 존재하는지 확인하고 가져오기
                for i, col in enumerate(columns):
                    if col['name'] == col_name and (not table_name or col['table'] == table_name):
                        matches.append(i)
                if len(matches) == 1:
                    return row[matches[0]], type
                #elif len(matches) > 1 -> ambiguous column
            return None, None

        def check_table_col(table_name, col_name, table_schemas):
            if table_name: #table reference가 존재하는 경우
                if table_name not in table_schemas:
                    print(f"{prompt}where clause trying to reference tables which are not specified")
                    return False
                if col_name not in table_schemas[table_name]:
                    print(f"{prompt}where clause trying to reference non existing column")
                    return False
            else: #table reference 없이 column만 있는 경우
                matched_tables = [t for t in table_schemas if col_name in table_schemas[t]]
                if len(matched_tables) == 0:
                    print(f"{prompt}where clause trying to reference non existing column")
                    return False
                elif len(matched_tables) > 1:
                    print(f"{prompt}where clause contains ambiguous column reference")
                    return False
            return True

        if hasattr(expr, 'data'):
            if expr.data == "comparison_predicate":
                left, op_token, right = expr.children
                left_val = None
                left_val_type = None
                right_val = None
                right_val_type = None
                
                if left.children[0].data == "column_reference":
                    t, c, _ = parse_table_col(left.children[0], columns)
                    if not check_table_col(t, c, table_schemas):
                        return False
                    else:
                        left_val, left_val_type = resolve_col_value(left)
                elif left.children[0].data == "comparable_value":
                    left_val = parse_comp_val(left.children[0])

                if right.children[0].data == "column_reference":
                    t, c, _= parse_table_col(right.children[0], columns)
                    if not check_table_col(t, c, table_schemas):
                        return False
                    else:
                        right_val, right_val_type = resolve_col_value(right)
                elif right.children[0].data == "comparable_value":
                    right_val = parse_comp_val(right.children[0])

                if left_val_type == None:
                    if right_val_type != None:
                        left_val = cast_value_to_column_type(right_val_type, left_val)
                if right_val_type == None:
                    if left_val_type != None:
                        right_val = cast_value_to_column_type(left_val_type, right_val)

                if left_val is None or right_val is None:
                    return False
                
                # 타입 추론
                def infer_type(val):
                    if isinstance(val, int):
                        return 'int'
                    elif isinstance(val, str):
                        if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
                            return 'date'
                        return 'char'
                    return None

                if left_val_type is None:
                    left_val_type = infer_type(left_val)

                if right_val_type is None:
                    right_val_type = infer_type(right_val)

                base_left = None
                base_right = None

                # 타입이 다르면 서로 비교 불가
                if left_val_type != right_val_type:
                    base_left = left_val_type.split('(')[0]
                    base_right = right_val_type.split('(')[0]
                    if base_left != base_right:
                        print(f"{prompt}Trying to compare incomparable columns or values")
                        return False
                    
                op = op_token.data

                # char type은 =, != 만 허용
                if base_left == "char":
                    if op not in ["equal", "notequal"]:
                        print(f"{prompt}Trying to compare incomparable columns or values")
                        return False

                if op == "equal":
                    return left_val == right_val
                elif op == "notequal":
                    return left_val != right_val
                elif op == "lessthan":
                    return left_val < right_val
                elif op == "greaterthan":
                    return left_val > right_val
                elif op == "lessequal":
                    return left_val <= right_val
                elif op == "greaterequal":
                    return left_val >= right_val
                else:
                    print(f"{prompt}Unsupportes operation: {op}")
            elif expr.data == "null_predicate":
                col_ref = expr.children[0]
                val, _ = resolve_col_value(col_ref)
                is_not = any(c is None for c in expr.children)
                return (val is not None) if is_not else (val is None)
            elif expr.data == "predicate":
                results = [check_expr(child) for child in expr.children if isinstance(child, Tree)]
                return all(results)

    try:
        check_expr_result = check_expr(boolean_term.children[1])
        if boolean_term.children[0] == None:
            return check_expr_result
        return not check_expr_result
    except Exception as e:
        print(f"{prompt}Where clause error: {str(e)}")
        return False
    
def where_clause_check(where_tree: Tree, table_schemas: dict, columns, row):
    if not where_tree or where_tree.data != "where_clause":
        return True
    
    boolean_expr = where_tree.children[1]
    children = boolean_expr.children
    if len(children) == 1:
        return predicate_check(children[0], table_schemas, columns, row)
    elif len(children) == 3:
        left_result = predicate_check(children[0], table_schemas, columns, row)
        operation = children[1].type
        right_result = predicate_check(children[2], table_schemas, columns, row)

        if operation == "OR":
            return left_result or right_result
        elif operation == "AND":
            return left_result and right_result
        else:
            print(f"WHERE: unsupported logical operator '{operation}'")
            return False
    else:
        print("WHERE: invalid expression structure (too many conditions)")
        return False

class MyTransformer(Transformer):
    def command(self, items):
        return items[0]
    
    def query_list(self, items):
        return items
    
    def query(self, items):
        return items[0]
    
    def create_table_query(self, items):
        table_name = items[2].children[0].value.lower()
        element_list = items[3].children
        # Remove LP and RP tokens
        for i in range(len(element_list)):
            if isinstance(element_list[i], Token) and element_list[i].type == 'LP':
                element_list.pop(i)
                break
        for i in range(len(element_list)-1, -1, -1):
            if isinstance(element_list[i], Token) and element_list[i].type == 'RP':
                element_list.pop(i)
                break
        
        # Filter out comma tokens
        element_list = [el for el in element_list if not (isinstance(el, Token) and el.value == ',')]
        
        key = table_name.encode("utf-8")

        # already exist table
        if metadata_db.exists(key):
            print(f"{prompt}Create table has failed: table with the same name already exists")
            return

        columns = []
        col_names = set()
        primary_keys = []
        foreign_keys = []

        # First pass: collect all column definitions
        for el in element_list:
            if el.data == 'table_element':
                node = el.children[0]
                if node.data == 'column_definition':
                    column_name = node.children[0].children[0].value.lower()
                    data_type_node = node.children[1]
                    
                    # Get data type string
                    dtype = get_data_type_str(data_type_node)

                    if dtype is None:
                        if data_type_node.children[0].type == 'TYPE_CHAR':
                            print(f"{prompt}Char length should be over 0")
                        else:
                            print(f"{prompt}Invalid data type")
                        return
                    
                    # Check for duplicate column names
                    if column_name in col_names:
                        print(f"{prompt}Create table has failed: column definition is duplicated")
                        return
                    
                    # Check NOT NULL constraint
                    not_null = False
                    for child in node.children[2:]:
                        if isinstance(child, Token) and child.type == 'NOT':
                            not_null = True
                            break
                    
                    col_names.add(column_name)
                    columns.append({"name": column_name, "type": dtype, "not_null": not_null})

        # Second pass: process constraints
        for el in element_list:
            if el.data == 'table_element':
                node = el.children[0]
                if node.data == 'table_constraint_definition':
                    constraint = node.children[0]
                    
                    if constraint.data == 'primary_key_constraint':
                        if primary_keys:
                            print(f"{prompt}Create table has failed: primary key definition is duplicated")
                            return
                        
                        # Extract primary key columns
                        pk_columns = constraint.children[2].children
                        # Remove LP and RP
                        pk_columns = [col for col in pk_columns if not (isinstance(col, Token) and (col.type == 'LP' or col.type == 'RP' or col.value == ','))]
                        
                        for pk_col_node in pk_columns:
                            pk_col = pk_col_node.children[0].value.lower()
                            if pk_col not in col_names:
                                print(f"{prompt}Create table has failed: cannot define non-existing column '{pk_col}' as primary key")
                                return
                            primary_keys.append(pk_col)
                    
                    elif constraint.data == 'referential_constraint':
                        # Extract foreign key columns
                        fk_columns_node = constraint.children[2]
                        fk_columns = []
                        for child in fk_columns_node.children:
                            if not isinstance(child, Token) or (child.type != 'LP' and child.type != 'RP' and child.value != ','):
                                if hasattr(child, 'children'):
                                    fk_columns.append(child.children[0].value.lower())
                        
                        # Extract referenced table
                        ref_table = constraint.children[4].children[0].value.lower()
                        
                        # Extract referenced columns
                        ref_columns_node = constraint.children[5]
                        ref_columns = []
                        for child in ref_columns_node.children:
                            if not isinstance(child, Token) or (child.type != 'LP' and child.type != 'RP' and child.value != ','):
                                if hasattr(child, 'children'):
                                    ref_columns.append(child.children[0].value.lower())
                        
                        # Validate foreign key columns
                        for fk_col in fk_columns:
                            if fk_col not in col_names:
                                print(f"{prompt}Create table has failed: cannot define non-existing column '{fk_col}' as foreign key")
                                return
                        
                        foreign_keys.append((fk_columns, ref_table, ref_columns))

        # Validate foreign key references
        for fk_cols, ref_table, ref_cols in foreign_keys:
            ref_key = ref_table.encode("utf-8")
            
            if not metadata_db.exists(ref_key):
                print(f"{prompt}Create table has failed: foreign key references non existing table")
                return
            
            ref_table_schema = json.loads(metadata_db.get(ref_key).decode("utf-8"))
            ref_column_dict = {col['name']: col for col in ref_table_schema}
            
            # Check if referenced columns exist and are primary keys
            for ref_col in ref_cols:
                if ref_col not in ref_column_dict:
                    print(f"{prompt}Create table has failed: foreign key references non existing column")
                    return
                
                if not ref_column_dict[ref_col].get('pk', False):
                    print(f"{prompt}Create table has failed: foreign key references non primary key column")
                    return
            
            # Check for type compatibility between foreign key and referenced columns
            for i, fk_col in enumerate(fk_cols):
                fk_col_type = next((col['type'] for col in columns if col['name'] == fk_col), None)
                ref_col_type = ref_column_dict[ref_cols[i]]['type']
                
                if fk_col_type != ref_col_type:
                    print(f"{prompt}Create table has failed: foreign key references wrong type")
                    return

        # Apply primary key constraints
        for pk_col in primary_keys:
            for col in columns:
                if col['name'] == pk_col:
                    col['pk'] = True
                    col['not_null'] = True  # Primary keys are implicitly NOT NULL
        
        # Apply foreign key constraints
        for fk_cols, ref_table, ref_cols in foreign_keys:
            for i, fk_col in enumerate(fk_cols):
                for col in columns:
                    if col['name'] == fk_col:
                        if 'fk' not in col:
                            col['fk'] = []
                        col['fk'].append({"ref_table": ref_table, "ref_col": ref_cols[i]})

        # Store the schema metadata
        metadata_db.put(key, json.dumps(columns).encode("utf-8"))
        
        # Create table database file
        table_db = db.DB(env)
        table_db.open(f"{table_name}.db", None, db.DB_HASH, db.DB_CREATE)
        table_db.close()
        
        print(f"{prompt}'{table_name}' table is created")

    def drop_table_query(self, items):
        table_name = items[2].children[0].value.lower()
        key = table_name.encode("utf-8")

        if not metadata_db.exists(key):
            print(f"{prompt}Drop table has failed: no such table")
            return

        # Check if this table is referenced by foreign keys in other tables
        for db_key, db_value in metadata_db.items():
            if db_key == key:
                continue
            
            other_table_schema = json.loads(db_value.decode("utf-8"))
            referenced = False
            
            for col in other_table_schema:
                if 'fk' in col:
                    if isinstance(col['fk'], list):
                        for fk in col['fk']:
                            if fk['ref_table'] == table_name:
                                referenced = True
                                break
                    else:
                        if col['fk']['ref_table'] == table_name:
                            referenced = True
                
                if referenced:
                    break
            
            if referenced:
                print(f"{prompt}Drop table has failed: '{table_name}' is referenced by another table")
                return

        # Drop the table
        metadata_db.delete(key)
        try:
            os.remove(os.path.join(DB_DIR, f"{table_name}.db"))
        except FileNotFoundError:
            pass
        
        print(f"{prompt}'{table_name}' table is dropped")

    def explain_query(self, items):
        table_name = items[1].children[0].value.lower()
        key = table_name.encode("utf-8")

        if not metadata_db.exists(key):
            print(f"{prompt}Explain table has failed: no such table")
            return

        columns = json.loads(metadata_db.get(key).decode("utf-8"))

        print("-----------------------------------------------------------------")
        print("column_name         | type      | null        | key")
        for col in columns:
            name = col['name'].ljust(20)
            dtype = col['type'].ljust(10)
            nullable = "N" if col.get('not_null', False) else "Y"
            nullable = nullable.ljust(12)
            
            key_types = []
            if col.get('pk', False):
                key_types.append("PRI")
            if col.get('fk'):
                key_types.append("FOR")
            
            key_str = "/".join(key_types)
            
            print(f"{name}| {dtype}| {nullable}| {key_str}")
        
        print("-----------------------------------------------------------------")
        print(f"{len(columns)} rows in set")

    def desc_query(self, items):
        return self.explain_query(items)

    def describe_query(self, items):
        return self.explain_query(items)
    
    def delete_query(self, items):
        table_name = items[2].children[0].value.lower()
        table_key = table_name.encode("utf-8")

        if not metadata_db.exists(table_key):
            print(f"{prompt}Delete has failed: no such table")
            return
        
        # Load table schema
        schema = json.loads(metadata_db.get(table_key).decode("utf-8"))
        table_schema_dict = {col['name']: col for col in schema}

        #Load rows from table
        table_db = db.DB(env)
        table_db.open(f"{table_name}.db", None, db.DB_HASH, db.DB_CREATE)
        cursor = table_db.cursor()

        #Collect rows to delete
        rows_to_delete = []
        record = cursor.first()

        #where 절 검사
        where_clause = None
        for item in items:
            if isinstance(item, Tree) and item.data == 'where_clause':
                where_clause = item
                break

        while record:
            key, value = record
            row_data = json.loads(value.decode("utf-8"))

            if where_clause == None:
                rows_to_delete.append(key)
                record = cursor.next()
                continue

            #column 정보 구성
            columns = [{**col, "table": table_name} for col in schema]

            if where_clause_check(where_clause, {table_name: table_schema_dict}, columns, row_data):
                rows_to_delete.append(key)
            

            record = cursor.next()
        cursor.close()

        # 다른 테이블에서 foreigh key로 참조하고 있는 튜플이 삭제 대상에 포함될 경우
        # 실제 삭제 수행 전에 foreign key 제약 조건 확인
        not_deleted_count = 0
        actually_deleted_count = 0

        for key in rows_to_delete:
            row_data = json.loads(table_db.get(key).decode("utf-8"))
            pk_values = [row_data[i] for i, col in enumerate(schema) if col.get('pk', False)]

            # 참조 여부 확인
            is_referenced = False

            for ref_table_key in metadata_db.keys():
                ref_table_name = ref_table_key.decode("utf-8")
                if ref_table_name == table_name:
                    continue

                ref_schema = json.loads(metadata_db.get(ref_table_key).decode("utf-8"))
                fk_columns = [
                    (i, col) for i, col in enumerate(ref_schema)
                    if col.get('fk', False) and col.get('ref') == table_name
                ]

                if not fk_columns:
                    continue

                if fk_columns:
                    print("찾았다")

                # 참조 테이블 열기
                ref_db = db.DB(env)
                ref_db.open(f"{ref_table_name}.db", None, db.DB_HASH, db.DB_CREATE)
                ref_cursor = ref_db.cursor()
                ref_record = ref_cursor.first()
                while ref_record:
                    _, ref_value = ref_record
                    ref_row = json.loads(ref_value.decode("utf-8"))
                    for fk_index, fk_col in fk_columns:
                        if ref_row[fk_index] in pk_values:
                            is_referenced = True
                            break
                    if is_referenced:
                        break
                    ref_record = ref_cursor.next()
                ref_cursor.close()
                ref_db.close()

                if is_referenced:
                    break

            if is_referenced:
                not_deleted_count += 1
            else:
                table_db.delete(key)
                actually_deleted_count += 1

        table_db.close()

        if (actually_deleted_count > 0) or  (actually_deleted_count == 0 and not_deleted_count == 0):
            print(f"{prompt}{actually_deleted_count} row(s) are deleted")

        if not_deleted_count > 0:
            print(f"{prompt}{not_deleted_count} row(s) are not deleted due to referential integrity")

    def show_query(self, items):
        print("------------------------")
        table_count = 0
        for k in metadata_db.keys():
            table_name = k.decode("utf-8")
            print(table_name)
            table_count += 1
        print("------------------------")
        print(f"{table_count} rows in set")

    def insert_query(self, items):
        table_name = items[2].children[0].children[0].value.lower()
        table_key = table_name.encode("utf-8")
        table_columns = items[2].children[1]
        
        if not metadata_db.exists(table_key):
            print(f"{prompt}Insert has failed: no such table")
            return
        
        # Get schema information
        columns = json.loads(metadata_db.get(table_key).decode("utf-8"))
        
        # Get values from query
        values_node = items[4].children
        values = []
        for child in values_node:
            if not isinstance(child, Token) or (child.type != 'LP' and child.type != 'RP' and child.value != ','):
                if hasattr(child, 'children') and len(child.children) > 0:
                    value = child.children[0].value

                    # Strip quotes from string values
                    if value.startswith('"') or value.startswith("'"):
                        value = value[1:-1]
                    else:
                        # Try to convert to int if possible
                        try:
                            if value.isdigit():
                                value = int(value)
                        except (ValueError, AttributeError):
                            pass
                            
                    values.append(value)

        # primary key uniqueness 검사
        pk_indices = [i for i, col in enumerate(columns) if col.get('pk', False)]

        if pk_indices:
            table_db = db.DB(env)
            table_db.open(f"{table_name}.db", None, db.DB_HASH, db.DB_CREATE)
            cursor = table_db.cursor()
            record = cursor.first()
            while record:
                _, value = record
                existing_row = json.loads(value.decode("utf-8"))
                #primary 값 비교
                if all(existing_row[i] == values[i] for i in pk_indices):
                    print(f"{prompt}Insert has failed: duplicate primary key")
                    table_db.close()
                    return
                record = cursor.next()
            cursor.close()
            table_db.close()

        # Validate number of values matches schema
        if len(values) != len(columns):
            print(f"{prompt}Insert has failed: types are not matched")
            return
        
        # 지정된 컬럼과 값의 개수가 다른 경우
        if table_columns:
            column_names = [child.children[0].value.lower() for child in table_columns.children
                            if hasattr(child, 'data') and child.data == 'column_name']
            if len(column_names) != len(values):
                print(f"{prompt}Insert has failed: types are not matched")
                return

            #존재하지 않는 column에 값을 삽입하는 경우
            schema_column_names = [col['name'] for col in columns]
            for col_name in column_names:
                if col_name not in schema_column_names:
                    print(f"{prompt}Insert has failed: '{col_name}' does not exist")
                    return
        
        # Validate NOT NULL constraints
        for i, col in enumerate(columns):
            if col.get('not_null', False) and (i >= len(values) or values[i] is None or values[i] == 'NULL'):
                print(f"{prompt}Insert has failed: '{col.get('name')}' is not nullable")
                return
            
        # value type 검사
        for i, col in enumerate(columns):
            expected_type = col['type'].lower()

            #INT
            if expected_type == 'int':
                if not isinstance(values[i], int):
                    print(f"{prompt}Insert has failed: types are not matched")
                    return
                
            #CHAR(n)
            elif expected_type.startswith('char('):
                if not isinstance(values[i], str):
                    print(f"{prompt}Insert has failed: types are not matched")
                    return
            
            #DATE
            elif expected_type == 'date':
                if not isinstance(values[i], str):
                    print(f"{prompt}Insert has failed: types are not matched")
                    return
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', values[i]):
                    print(f"{prompt}Insert has failed: types are not matched")
                    return

        # Truncate char values if needed
        for i, col in enumerate(columns):
            if col['type'].startswith('char(') and isinstance(values[i], str):
                match = re.match(r'char\((\d+)\)', col['type'])
                if match:
                    max_length = int(match.group(1))
                    values[i] = values[i][:max_length]
        
        # Store the row
        table_db = db.DB(env)
        table_db.open(f"{table_name}.db", None, db.DB_HASH, db.DB_CREATE)
        
        # Generate a unique row ID (using timestamp or sequence)
        row_id = str(len(list(table_db.keys())) + 1).encode("utf-8")
        
        table_db.put(row_id, json.dumps(values).encode("utf-8"))
        table_db.close()
        
        print(f"{prompt}1 row inserted")

    def select_query(self, items):
        select_list = None
        from_clause = None
        where_clause = None
        order_by_clause = None

        column_list = [] # select list에 나오는 column
        from_table_name_list = []

        table_schemas = {}
        table_rows = {}

        for item in items:
            if hasattr(item, 'data'):
                if item.data == 'table_expression':
                    for child in item.children:
                        if hasattr(child, 'data'):
                            if child.data == 'from_clause':
                                from_clause = child
                            elif child.data == 'where_clause':
                                where_clause = child
                            elif child.data == 'order_by':
                                order_by_clause = child
                elif item.data == 'select_list':
                    select_list = item

        #column_list from select list
        for col in select_list.children:
            if isinstance(col, Token) and col.type == 'WILDCARD':
                column_list.append({'table_name': None, 'column_name': '*'})
            elif isinstance(col, Tree) and col.data == 'selected_column':
                children = col.children
                if children[0] == None:
                    #table name 생략
                    column_name = children[1].children[0].value.lower()
                    column_list.append({'table_name' : None, 'column_name': column_name})
                elif hasattr(children[0], 'data') and children[0].data == 'table_name':
                    #table_name.column_name
                    table_name = children[0].children[0].value.lower()
                    column_name = children[1].children[0].value.lower()
                    column_list.append({'table_name': table_name, 'column_name': column_name})

        # from clause
        table_ref_list = None
        for child in from_clause.children:
            if hasattr(child, 'data') and child.data == 'table_reference_list':
                table_ref_list = child
                break
        
        if not table_ref_list:
            print(f"{prompt}Syntax error")
            return
        from_table_name_list, join_clauses_list = extract_from_tables(table_ref_list)

        # from_clause에 있는 table_name 존재 여부 검사
        for table_name in from_table_name_list:
            if not metadata_db.exists(table_name.encode("utf-8")):
                print(f"{prompt}Select has failed: {table_name} does not exist")
                return
            

        # from_table_name_list에서 각 table의 schema, data 가져오기
        for table_name in from_table_name_list:
            
            #schema load
            schema = json.loads(metadata_db.get(table_name.encode("utf-8")).decode("utf-8"))
            table_schemas[table_name] = {col['name']: col for col in schema}

            #db data load
            table_db = db.DB(env)
            table_db.open(f"{table_name}.db", None, db.DB_HASH, db.DB_RDONLY)
            rows = []
            cursor = table_db.cursor()
            record = cursor.first()
            while record:
                _, value = record
                rows.append(json.loads(value.decode("utf-8")))
                record = cursor.next()
            cursor.close()
            table_db.close()
            table_rows[table_name] = rows


        # column list 채우기
        # column list의 table_name이 있는 경우 -> table_name이 from_table_name_list에 있는 지 확인하고 column들이 table_name에 있는지 검사
        # column list의 table_name이 없는 경우 -> table_schema에서 찾아서 넣기
        # whid card -> 전체 table

        new_column_list = []

        for col in column_list:
            table_name = col['table_name']
            column_name = col['column_name']

            if column_name == "*":
                for selected_tname in from_table_name_list:
                    for col_name in table_schemas[selected_tname]:
                        new_column_list.append({
                            'table_name': selected_tname,
                            'column_name': col_name
                        })
                continue

            # table_name 있는 경우
            if table_name:
                if table_name not in from_table_name_list:
                    print(f"{prompt}Select has failed: '{table_name}' does not exist")
                    return

                if column_name not in table_schemas[table_name]:
                    print(f"{prompt}Select has failed: fail to resolve '{column_name}'")
                    return
                
                new_column_list.append(col)
            
            #table name 생략된 경우
            else:
                found_table = None
                for selected_tname in from_table_name_list:
                    if column_name in table_schemas[selected_tname]:
                        if found_table:
                            print(f"{prompt}Select has failed: fail to resolve '{column_name}'")
                            return
                        found_table = selected_tname

                if not found_table:
                    print(f"{prompt}Select has failed: fail to resolve '{column_name}'")
                    return
                
                col['table_name'] = found_table
                new_column_list.append(col)
        column_list = new_column_list

        #join function
        def inner_join(tables_name_list, join_clauses, table_rows):

            base_table = tables_name_list[0]
            base_columns = list(table_schemas[base_table].keys())

            result = [
                {f"{base_table}.{col}": val for col, val in zip(base_columns, row)} 
                for row in table_rows[base_table]
            ]


            for join_clause in join_clauses:
                join_table_name = join_clause.children[1].children[0].value
                join_condition = join_clause.children[3]

                left_table = join_condition.children[0].children[0].value
                left_column = join_condition.children[1].children[0].value
                right_table = join_condition.children[2].children[0].value
                right_column = join_condition.children[3].children[0].value

                right_rows_raw = table_rows[join_table_name]
                right_schema = list(table_schemas[join_table_name].keys())
                right_rows = [
                    {col: val for col, val in zip(right_schema, row)}
                    for row in right_rows_raw
                ]

                new_result = []
                for left_row in result:
                    left_value = left_row[f"{left_table}.{left_column}"]
                    for right_row in right_rows:
                        if right_row[right_column] == left_value:
                            combined_row = left_row.copy()
                            for k, v in right_row.items():
                                combined_row[f"{join_table_name}.{k}"] = v
                            new_result.append(combined_row)

                result = new_result
            return result

        #join operation
        if join_clauses_list:
            result = inner_join(from_table_name_list, join_clauses_list, table_rows)
        else:
            #단일 테이블일 경우
            only_table = from_table_name_list[0]
            columns = list(table_schemas[only_table].keys())
            result = [{f"{only_table}.{col}": val for col, val in zip(columns, row)} for row in table_rows[only_table]]

        # where clause 
        columns = []
        for table_name in from_table_name_list:
            for col_name, col_meta in table_schemas[table_name].items():
                columns.append({
                    'table': table_name,
                    'name': col_name,
                    'type': col_meta['type']
                })

        filtered_rows = []
        for row_dict in result:
            row = [row_dict.get(f"{col['table']}.{col['name']}") for col in columns]
            if where_clause_check(where_clause, table_schemas, columns, row):
                filtered_rows.append(row_dict)

        #order by
        if order_by_clause:
            order_targets = []
            for child in order_by_clause.children[2:]:
                if hasattr(child, 'data') and child.data == 'order_item':
                    col_name = child.children[0].children[0].value.lower()
                    order = 'asc'
                    if child.children[1]:
                        order = child.children[1].type.lower()
                    
                    table_found = None
                    for t in from_table_name_list:
                        if col_name in table_schemas[t]:
                            if table_found:
                                print(f"{prompt}#ORDER BY clause contains ambiguous column reference")
                                return
                            table_found = t
                            order_targets.append((f"{table_found}.{col_name}", order))
                    
                    #정렬 실행
                    for col_key, order_type in reversed(order_targets):
                        filtered_rows.sort(key = lambda row: row.get(col_key), reverse = (order_type == "desc"))

        # display result
        output_indices = []
        output_col_names = []

        if any(col['column_name'] == '*' for col in column_list):
            for i, col_def in enumerate(columns):
                output_indices.append(i)
                output_col_names.append(f"{col_def['table']}.{col_def['name']}".upper())
        else:
            for col in column_list:
                for i, col_def in enumerate(columns):
                    if col_def['table'] == col['table_name'] and col_def['name'] == col['column_name']:
                        output_indices.append(i)
                        output_col_names.append(f"{col['table_name']}.{col['column_name']}".upper())
                        break

        col_widths = [
            max(len(name), max(len(str(row.get(f"{columns[i]['table']}.{columns[i]['name']}", ""))) for row in filtered_rows) if filtered_rows else 0) + 2
            for i, name in zip(output_indices, output_col_names)
        ]

        print("---------------------------------------")
        print("|".join(name.center(col_widths[i]) for i, name in enumerate(output_col_names)))
        for row in filtered_rows:
            print("|".join(str(row.get(f"{columns[i]['table']}.{columns[i]['name']}", "")).center(col_widths[j])
                        for j, i in enumerate(output_indices)))
        print("---------------------------------------")
        print(f"{len(filtered_rows)} rows in set")

    def rename_query(self, items):
        old_name = items[1].children[0].children[0].children[0].value.lower()
        new_name = items[1].children[0].children[2].children[0].value.lower()
        old_key = old_name.encode("utf-8")
        new_key = new_name.encode("utf-8")
        
        if not metadata_db.exists(old_key):
            print(f"{prompt}Rename table has failed: no such table")
            return
        
        if metadata_db.exists(new_key) or (old_name == new_name):
            print(f"{prompt}Rename table has failed: new table name already exists")
            return
        
        # Update references in other tables
        for db_key, db_value in metadata_db.items():
            if db_key == old_key:
                continue
            
            other_table_schema = json.loads(db_value.decode("utf-8"))
            updated = False
            
            for col in other_table_schema:
                if 'fk' in col:
                    if isinstance(col['fk'], list):
                        for fk in col['fk']:
                            if fk['ref_table'] == old_name:
                                fk['ref_table'] = new_name
                                updated = True
                    else:
                        if col['fk']['ref_table'] == old_name:
                            col['fk']['ref_table'] = new_name
                            updated = True
            
            if updated:
                metadata_db.put(db_key, json.dumps(other_table_schema).encode("utf-8"))
        
        # Move schema information
        table_schema = metadata_db.get(old_key)
        metadata_db.delete(old_key)
        metadata_db.put(new_key, table_schema)
        
        # Rename database file
        try:
            os.rename(os.path.join(DB_DIR, f"{old_name}.db"), os.path.join(DB_DIR, f"{new_name}.db"))
        except FileNotFoundError:
            # If database file doesn't exist yet
            print("rename has failed: no such table")
            return
        
        # open and close after rename
        try:
            table_db = db.DB(env)
            table_db.open(f"{new_name}.db", None, db.DB_HASH, db.DB_CREATE)
            table_db.close()
        except Exception as e:
            print(f"Warning: renamed table could not be reopened: {e}")
            return
        
        print(f"{prompt}'{new_name}' is renamed")

    def truncate_query(self, items):
        table_name = items[1].children[0].value.lower()
        table_key = table_name.encode("utf-8")
        
        if not metadata_db.exists(table_key):
            print(f"{prompt}Truncate table has failed: no such table")
            return
        
        # Check if this table is referenced by foreign keys in other tables
        for db_key, db_value in metadata_db.items():
            if db_key == table_key:
                continue
            
            other_table_schema = json.loads(db_value.decode("utf-8"))
            referenced = False
            
            for col in other_table_schema:
                if 'fk' in col:
                    if isinstance(col['fk'], list):
                        for fk in col['fk']:
                            if fk['ref_table'] == table_name:
                                referenced = True
                                break
                    else:
                        if col['fk']['ref_table'] == table_name:
                            referenced = True
                
                if referenced:
                    break
            
            if referenced:
                print(f"{prompt}Truncate has failed: '{table_name}' is referenced by another table")
                return
        
        # Truncate table data
        table_db = db.DB(env)
        table_db.open(f"{table_name}.db", None, db.DB_HASH, db.DB_CREATE)
        
        # Delete all records
        cursor = table_db.cursor()
        rec = cursor.first()
        while rec:
            key, _ = rec
            table_db.delete(key)
            rec = cursor.next()
        cursor.close()
        table_db.close()
        
        print(f"{prompt}'{table_name}' is truncated")

    def exit_query(self, items):
        metadata_db.close()
        env.close()
        sys.exit()
    
        

# 사용자 입력 루프
def main():
    while True:
        buffer = ""
        
        while True:
            print(prompt, end="")
            try:
                line = input()
            except EOFError:
                metadata_db.close()
                env.close()
                sys.exit()
            
            if line.strip() == "" and ";" not in buffer:
                continue
            
            buffer += line + " "
            
            if ";" in line:
                break
        
        queries = [q.strip() + ";" for q in buffer.split(";") if q.strip()]
        
        for query in queries:
            try:
                parsed = sql_parser.parse(query)
                MyTransformer().transform(parsed)
            except Exception as e:
                print(f"{prompt}Syntax error : {e}")


if __name__ == "__main__":
    main()
