# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:59:32 2017

@author: Tyler Hughes
"""

import sqlalchemy as sqla

user = 'user'
password = 'password'  # getpass.getpass()
server = 'server'
database = 'database'

# The model needs some information about the schema of the dataset in order to work properly.
current_state_field = 'pf_currentstate'
next_state_field = 'pf_nextstate'
state_duration_field = 'pf_stateduration'
key_field = 'pf_enc'
predictive_variables = ['DoctorNameLast', 'LocationCode', 'VisitType']  # , 'MonthName', 'DayOfWeek', 'HourOfDay']
absorbing_state = 'Check-out' # name the absorbing state


def connect_to_data():
    engine = sqla.create_engine(
        'mssql+pymssql://' + user + ':' + password + '@' + server + '/' + database
    )
    print('\nSuccessfully connected to %s as %s!' % (database,user))
    return engine.connect()

# Put your SQL query for base data set here
pf_query = """
SELECT
	FlowStateMatchup.pf_enc,
	PatientFlowFeatures.LocationCode,
	PatientFlowFeatures.DoctorNameLast,
	PatientFlowFeatures.VisitType,
	FlowStateMatchup.pf_currentstate,
	FlowStateMatchup.pf_nextstate,
	FlowStateMatchup.pf_stateduration
FROM
(
SELECT
	CurrentFlowState.pfid AS pfid,
	CurrentFlowState.pf_enc,
	CurrentFlowState.pf_state AS pf_currentstate,
	CASE CurrentFlowState.pf_state
		WHEN 'Check-out'
		THEN 'Check-out'
		ELSE COALESCE(NextFlowState.pf_state, 'Check-out') 
		END
	AS pf_nextstate,
	CurrentFlowState.pf_time,
	DATEDIFF(mi, CurrentFlowState.pf_time, COALESCE(NextFlowState.pf_time, CurrentFlowState.pf_time)) AS pf_stateduration,
	ROW_NUMBER() OVER (
				PARTITION BY CurrentFlowState.pf_enc,
				CASE CurrentFlowState.pf_state
					WHEN 'Check-out'
					THEN 'Check-out'
					ELSE COALESCE(NextFlowState.pf_state, 'Check-out') 
					END,
				CurrentFlowState.pf_time
				ORDER BY CurrentFlowState.pf_time) 
	AS uniqueness 
FROM
(
SELECT
	flow.Id AS pfid,
	flow.EncounterNumber AS pf_enc,
	pftype.[Name] AS pf_state,
	flow.EventTime AS pf_time,
	DENSE_RANK() OVER (PARTITION BY flow.EncounterNumber ORDER BY flow.EventTime ASC) AS pf_statehierarchy
FROM
	Facts.EncounterPatientFlow flow
INNER JOIN
	Dimensions.PatientFlowType pftype
ON
	flow.PatientFlowType_Id = pftype.Id
) AS CurrentFlowState
LEFT JOIN
(
SELECT
	flow.Id AS pfid,
	flow.EncounterNumber AS pf_enc,
	pftype.[Name] AS pf_state,
	flow.EventTime AS pf_time,
	DENSE_RANK() OVER (PARTITION BY flow.EncounterNumber ORDER BY flow.EventTime ASC) - 1 AS pf_statehierarchy
FROM
	Facts.EncounterPatientFlow flow
INNER JOIN
	Dimensions.PatientFlowType pftype
ON
	flow.PatientFlowType_Id = pftype.Id
) AS NextFlowState
ON
	CurrentFlowState.pf_enc = NextFlowState.pf_enc
	AND CurrentFlowState.pf_statehierarchy = NextFlowState.pf_statehierarchy
WHERE 
	YEAR(CurrentFlowState.pf_time) >= 2016
AND
	(
	CurrentFlowState.pf_state = 'Check-out' 
	OR 
	CurrentFlowState.pf_state != NextFlowState.pf_state
	)
GROUP BY
	CurrentFlowState.pfid,
	CurrentFlowState.pf_enc,
	CurrentFlowState.pf_state,
	CASE CurrentFlowState.pf_state
		WHEN 'Check-out'
		THEN 'Check-out'
		ELSE COALESCE(NextFlowState.pf_state, 'Check-out') 
		END,
	CurrentFlowState.pf_time,
	DATEDIFF(mi, CurrentFlowState.pf_time, COALESCE(NextFlowState.pf_time, CurrentFlowState.pf_time))
) FlowStateMatchup
INNER JOIN
    (
	 SELECT DISTINCT EncounterNumber, COUNT(DISTINCT VisitType) AS num_visit_types, COUNT(pf_state) AS both_arrived_and_checkout
		FROM
		(
		SELECT DISTINCT EncounterNumber, pftype.[Name] as pf_state, vis.[Name] AS VisitType
		 FROM Facts.EncounterPatientFlow flow
		INNER JOIN
		Dimensions.PatientFlowType pftype
		ON
		flow.PatientFlowType_Id = pftype.Id
		INNER JOIN
		Dimensions.VisitType vis
		ON
		flow.VisitType_Id = vis.Id
		WHERE pftype.Name = 'Arrived' OR pftype.Name = 'Check-out'
		) adequate_samples
		GROUP BY EncounterNumber
		HAVING COUNT(pf_state) = 2 AND COUNT(DISTINCT VisitType) = 1
		) encounter_criteria
	ON FlowStateMatchup.pf_enc = encounter_criteria.EncounterNumber
INNER JOIN
	(
	SELECT
		pf.Id AS pfid,
		doc.DoctorNameLast,
		vis.[Name] AS VisitType,
		loc.LocationCode,
       time.MonthName,
       time.DayOfWeek,
       DATEPART(hour, time.DateTimeStamp) AS HourOfDay
	FROM
		Facts.EncounterPatientFlow pf
	INNER JOIN
		Dimensions.Doctor doc
	ON
		pf.Doctor_Id = doc.Id
	INNER JOIN
		Dimensions.VisitType vis
	ON
		pf.VisitType_Id = vis.Id
	INNER JOIN
		Dimensions.LocationPod loc
	ON
		pf.LocationPod_Id = loc.Id
   INNER JOIN
       Dimensions.TimePeriod time
    ON pf.Date_Id = time.Id
	GROUP BY
        pf.Id,
		doc.DoctorNameLast,
		vis.[Name],
		loc.LocationCode,
       time.MonthName,
       time.DayOfWeek,
       DATEPART(hour, time.DateTimeStamp)
	) PatientFlowFeatures
ON
	FlowStateMatchup.pfid = PatientFlowFeatures.pfid
WHERE 
	uniqueness=1
"""
