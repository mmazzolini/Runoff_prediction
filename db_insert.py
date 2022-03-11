import psycopg2
import numpy as np



def insert(STAT_CODE, df):
    conn = psycopg2.connect(host="10.8.244.31",
                       database="climate_data",
                       user="ado_admin",
                       password="oda347hydro",
                       port=5432)
    cur = conn.cursor()
    

    for i in df.index:
 
        query = f"""
                INSERT INTO "ML_discharge"."mod_disc"("id_station", "date", "discharge", "meas_disch_presence") 
                VALUES ('{STAT_CODE}', '{np.datetime64(i,'D')}', '{df.loc[i].prediction}', '{df.loc[i].meas_disch_presence}'); 
                """
        cur.execute(query)
        
    conn.commit()

    # close the connection when finished
    cur.close()
    conn.close()
    
    

def insert_pred(STAT_CODE, df):
    conn = psycopg2.connect(host="10.8.244.31",
                       database="climate_data",
                       user="ado_admin",
                       password="oda347hydro",
                       port=5432)
    cur = conn.cursor()
    

    for i in df.index:
 
        query = f"""
                INSERT INTO "ML_discharge"."pred_disch"("id_station", "date", "10_d_disch", "20_d_disch")
                VALUES ('{STAT_CODE}', '{np.datetime64(i,'D')}', '{df.loc[i,'10']}', '{df.loc[i,'20']}'); 
                """
        cur.execute(query)
        
    conn.commit()

    # close the connection when finished
    cur.close()
    conn.close()