CREATE TABLE temp
SELECT
  *
FROM
  (
    SELECT
      api.SUBJECT_ID,
      api.HADM_ID,
      api.ICUSTAY_ID,
      api.age,
      api.ETHNICITY,
      api.GENDER,
      TIMESTAMPDIFF(MINUTE, api.ADMITTIME, l.CHARTTIME) AS 'time',
      l.ITEMID,
      l.VALUENUM,
      l.VALUEUOM
    FROM
      (
        (
          SELECT
            ap.SUBJECT_ID,
            ap.HADM_ID,
            i.ICUSTAY_ID,
            ap.ADMITTIME,
            ap.age,
            ap.ETHNICITY,
            ap.GENDER
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
                        mimic3.ADMISSIONS
                    ) a
                    INNER JOIN (
                      SELECT
                        SUBJECT_ID,
                        DOB,
                        GENDER
                      FROM
                        mimic3.PATIENTS
                    ) p ON (a.SUBJECT_ID = p.SUBJECT_ID)
                  )
              ) ap
              LEFT JOIN (
                SELECT
                  SUBJECT_ID, HADM_ID, ICUSTAY_ID
                FROM
                  mimic3.ICUSTAYS
              ) i ON (ap.SUBJECT_ID = i.SUBJECT_ID)
              AND (ap.HADM_ID = i.HADM_ID)
            )
        ) api
        INNER JOIN (
          SELECT
            SUBJECT_ID,
            HADM_ID,
            CHARTTIME,
            ITEMID,
            VALUENUM,
            VALUEUOM
          FROM
            mimic3.LABEVENTS
          WHERE
            ITEMID = 50912
            AND VALUENUM IS NOT NULL
        ) l ON (api.HADM_ID = l.HADM_ID)
        AND (api.SUBJECT_ID = l.SUBJECT_ID)
      )
    WHERE
      api.age > 0
      AND api.age < 300
  ) apil
ORDER BY
  apil.SUBJECT_ID,
  apil.HADM_ID,
  apil.ITEMID,
  apil.time,
  apil.VALUENUM
