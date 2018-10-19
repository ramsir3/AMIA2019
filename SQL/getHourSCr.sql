SELECT COUNT(*)
FROM (
    SELECT r.SUBJECT_ID, r.HADM_ID, r.time, r.age, r.GENDER, r.ETHNICITY, r.VALUENUM, r.baseline_Scr, r.VALUENUM/r.baseline_Scr as 'scr_ratio'
    FROM
    (
        SELECT
            scr_info.SUBJECT_ID,
            scr_info.HADM_ID,
            scr_info.ITEMID,
            scr_info.time,
            scr_info.age,
            scr_info.GENDER,
            scr_info.ETHNICITY,
            scr_info.VALUENUM,
            scr_info.VALUEUOM,
            (
                CASE
                WHEN (scr_info.ETHNICITY LIKE 'BLACK%') AND (scr_info.GENDER = 'F') THEN ROUND(POW(75/(0.742*1.21*186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
                WHEN scr_info.GENDER = 'F' THEN ROUND(POW(75/(0.742*186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
                WHEN scr_info.ETHNICITY LIKE 'BLACK%' THEN ROUND(POW(75/(1.21*186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
                ELSE ROUND(POW(75/(186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
                END
            ) AS 'baseline_Scr'
        FROM
        (
            SELECT
                t.SUBJECT_ID,
                t.HADM_ID,
                t.ITEMID,
                t.age,
                t.ETHNICITY,
                t.GENDER,
                t.time,
                t.VALUENUM,
                t.VALUEUOM
            FROM
                (
                SELECT
                    ap.SUBJECT_ID,
                    ap.HADM_ID,
                    ap.age,
                    ap.ETHNICITY,
                    ap.GENDER,
                    TIMESTAMPDIFF(MINUTE, ap.ADMITTIME, l.CHARTTIME) AS 'time',
                    l.ITEMID,
                    l.VALUENUM,
                    l.VALUEUOM
                FROM
                    (
                        (
                            SELECT
                                a.SUBJECT_ID,
                                a.HADM_ID,
                                a.ADMITTIME,
                                CAST(TIMESTAMPDIFF(YEAR, p.DOB, a.ADMITTIME) as int) AS 'age',
                                a.ETHNICITY,
                                p.GENDER
                            FROM
                                (
                                    (
                                        SELECT
                                            ADMITTIME,
                                            SUBJECT_ID,
                                            HADM_ID,
                                            ETHNICITY
                                        FROM
                                            ADMISSIONS
                                    ) a
                                    INNER JOIN (
                                        SELECT
                                            SUBJECT_ID,
                                            DOB,
                                            GENDER
                                        FROM
                                            PATIENTS
                                    ) p ON (a.SUBJECT_ID = p.SUBJECT_ID)
                                )
                        ) ap
                        INNER JOIN (
                            SELECT
                                SUBJECT_ID,
                                HADM_ID,
                                CHARTTIME,
                                ITEMID,
                                VALUENUM,
                                VALUEUOM
                            FROM
                                LABEVENTS
                            WHERE
                                ITEMID = 50912
                                AND VALUENUM IS NOT NULL
                        ) l ON (ap.HADM_ID = l.HADM_ID)
                        AND (ap.SUBJECT_ID = l.SUBJECT_ID)
                    )
                WHERE ap.age > 0 AND ap.age < 300
                ) t
            ORDER BY
            t.SUBJECT_ID,
            t.HADM_ID,
            t.ITEMID,
            t.time,
            t.VALUENUM
        ) scr_info
    ) r
    WHERE r.VALUENUM/r.baseline_Scr > 1.5
) allrows