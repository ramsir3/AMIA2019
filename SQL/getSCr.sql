CREATE TABLE temp
SELECT
  ap.SUBJECT_ID,
  ap.HADM_ID,
  ap.age,
  ap.ETHNICITY,
  ap.GENDER,
  TIMESTAMPDIFF(MINUTE, ap.ADMITTIME, l.CHARTTIME) AS 'time',
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
    ) l ON (ap.HADM_ID = l.HADM_ID)
    AND (ap.SUBJECT_ID = l.SUBJECT_ID)
  )
WHERE
  ap.age >= 18
  AND ap.age < 300
ORDER BY
  ap.SUBJECT_ID,
  ap.HADM_ID,
  time,
  l.VALUENUM
